#!/usr/bin/env python3
"""
TaskFlow: A zero-dependency CLI task & workflow engine with priority scoring,
dependency resolution, tag filtering, and JSON persistence.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_STORE = Path.home() / ".taskflow_tasks.json"

PRIORITY_WEIGHTS: Dict[str, int] = {
    "critical": 100,
    "high": 70,
    "medium": 40,
    "low": 15,
}

VALID_STATUSES = ("todo", "in-progress", "blocked", "done")


class TaskFlowError(Exception):
    """Domain-specific exception for TaskFlow operations."""


@dataclasses.dataclass
class Task:
    id: int
    title: str
    priority: str = "medium"
    status: str = "todo"
    tags: List[str] = dataclasses.field(default_factory=list)
    depends_on: List[int] = dataclasses.field(default_factory=list)
    due_date: Optional[str] = None
    created_at: str = dataclasses.field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )
    completed_at: Optional[str] = None

    def __post_init__(self) -> None:
        self.priority = self.priority.lower().strip()
        self.status = self.status.lower().strip()
        if self.priority not in PRIORITY_WEIGHTS:
            raise TaskFlowError(
                f"Invalid priority '{self.priority}'. "
                f"Choose from: {', '.join(PRIORITY_WEIGHTS.keys())}"
            )
        if self.status not in VALID_STATUSES:
            raise TaskFlowError(
                f"Invalid status '{self.status}'. "
                f"Choose from: {', '.join(VALID_STATUSES)}"
            )
        self.tags = sorted({t.strip().lower() for t in self.tags if t.strip()})
        self.depends_on = sorted({int(d) for d in self.depends_on})
        if self.due_date is not None:
            validate_iso_date(self.due_date)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        return cls(
            id=int(data["id"]),
            title=str(data["title"]),
            priority=data.get("priority", "medium"),
            status=data.get("status", "todo"),
            tags=data.get("tags", []),
            depends_on=data.get("depends_on", []),
            due_date=data.get("due_date"),
            created_at=data.get(
                "created_at", dt.datetime.now(dt.timezone.utc).isoformat()
            ),
            completed_at=data.get("completed_at"),
        )


def validate_iso_date(date_str: str) -> dt.date:
    try:
        return dt.date.fromisoformat(date_str)
    except ValueError as exc:
        raise TaskFlowError(
            f"Invalid date '{date_str}'. Expected YYYY-MM-DD format."
        ) from exc


class TaskStore:
    """Handles atomic JSON persistence and queries for Task objects."""

    def __init__(self, filepath: Path = DEFAULT_STORE) -> None:
        self.filepath = Path(filepath)
        self.tasks: Dict[int, Task] = {}
        self.load()

    def load(self) -> None:
        if not self.filepath.exists():
            self.tasks = {}
            return
        try:
            raw = self.filepath.read_text(encoding="utf-8")
            payload = json.loads(raw)
            if not isinstance(payload, dict) or "tasks" not in payload:
                raise TaskFlowError("Malformed task store JSON structure.")
            loaded: Dict[int, Task] = {}
            for item in payload["tasks"]:
                task = Task.from_dict(item)
                loaded[task.id] = task
            self.tasks = loaded
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise TaskFlowError(f"Failed to load task store: {exc}") from exc

    def save(self) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.filepath.with_suffix(".tmp")
        payload = {
            "version": 1,
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "tasks": [t.to_dict() for t in self.ordered_tasks()],
        }
        try:
            temp_path.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(temp_path, self.filepath)
        except OSError as exc:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise TaskFlowError(f"Failed to save task store: {exc}") from exc

    def next_id(self) -> int:
        return (max(self.tasks.keys()) + 1) if self.tasks else 1

    def ordered_tasks(self) -> List[Task]:
        return [self.tasks[k] for k in sorted(self.tasks.keys())]

    def get(self, task_id: int) -> Task:
        if task_id not in self.tasks:
            raise TaskFlowError(f"Task #{task_id} not found.")
        return self.tasks[task_id]

    def _has_cycle(self, start_id: int, candidate_deps: List[int]) -> bool:
        """Detects dependency cycles using DFS."""
        visited: set[int] = set()

        def dfs(current_id: int) -> bool:
            if current_id == start_id:
                return True
            if current_id in visited:
                return False
            visited.add(current_id)
            parent = self.tasks.get(current_id)
            if not parent:
                return False
            for dep in parent.depends_on:
                if dfs(dep):
                    return True
            return False

        for dep_id in candidate_deps:
            if dep_id == start_id or dfs(dep_id):
                return True
        return False

    def add_task(
        self,
        title: str,
        priority: str = "medium",
        tags: Optional[List[str]] = None,
        depends_on: Optional[List[int]] = None,
        due_date: Optional[str] = None,
    ) -> Task:
        clean_title = title.strip()
        if not clean_title:
            raise TaskFlowError("Task title cannot be empty.")

        task_id = self.next_id()
        deps = sorted({int(d) for d in (depends_on or [])})
        for dep_id in deps:
            if dep_id not in self.tasks:
                raise TaskFlowError(f"Dependency task #{dep_id} does not exist.")
        if self._has_cycle(task_id, deps):
            raise TaskFlowError("Circular dependency detected.")

        task = Task(
            id=task_id,
            title=clean_title,
            priority=priority,
            status="todo",
            tags=tags or [],
            depends_on=deps,
            due_date=due_date,
        )
        self.tasks[task.id] = task
        self.save()
        return task

    def unmet_dependencies(self, task: Task) -> List[int]:
        unmet: List[int] = []
        for dep_id in task.depends_on:
            dep_task = self.tasks.get(dep_id)
            if dep_task is None or dep_task.status != "done":
                unmet.append(dep_id)
        return unmet

    def urgency_score(self, task: Task, today: Optional[dt.date] = None) -> int:
        """Computes a dynamic urgency score based on priority, due date, and blockers."""
        if task.status == "done":
            return -1000

        ref_day = today or dt.date.today()
        score = PRIORITY_WEIGHTS[task.priority]

        if task.due_date:
            due = validate_iso_date(task.due_date)
            days_left = (due - ref_day).days
            if days_left < 0:
                score += 60 + min(abs(days_left) * 5, 40)
            elif days_left == 0:
                score += 45
            elif days_left <= 3:
                score += 25
            elif days_left <= 7:
                score += 10

        if self.unmet_dependencies(task):
            score -= 35

        return score

    def set_status(self, task_id: int, status: str) -> Task:
        task = self.get(task_id)
        normalized = status.lower().strip()
        if normalized not in VALID_STATUSES:
            raise TaskFlowError(
                f"Invalid status '{status}'. Choose from: {', '.join(VALID_STATUSES)}"
            )
        if normalized in ("in-progress", "done"):
            unmet = self.unmet_dependencies(task)
            if unmet:
                raise TaskFlowError(
                    f"Cannot mark task #{task_id} as '{normalized}'; "
                    f"uncompleted dependencies: {unmet}"
                )

        task.status = normalized
        if normalized == "done":
            task.completed_at = dt.datetime.now(dt.timezone.utc).isoformat()
        else:
            task.completed_at = None

        self.save()
        return task

    def delete_task(self, task_id: int) -> None:
        self.get(task_id)
        dependents = [
            t.id for t in self.tasks.values() if task_id in t.depends_on
        ]
        if dependents:
            raise TaskFlowError(
                f"Cannot delete task #{task_id}; required by tasks: {dependents}"
            )
        del self.tasks[task_id]
        self.save()

    def filter_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        tag: Optional[str] = None,
        ready_only: bool = False,
    ) -> List[Task]:
        results: List[Task] = []
        for task in self.ordered_tasks():
            if status and task.status != status.lower():
                continue
            if priority and task.priority != priority.lower():
                continue
            if tag and tag.lower() not in task.tags:
                continue
            if ready_only and (
                task.status == "done" or bool(self.unmet_dependencies(task))
            ):
                continue
            results.append(task)
        return results

    def summary_stats(self) -> Dict[str, Any]:
        total = len(self.tasks)
        by_status = {s: 0 for s in VALID_STATUSES}
        by_priority = {p: 0 for p in PRIORITY_WEIGHTS}
        overdue = 0
        today = dt.date.today()

        for task in self.tasks.values():
            by_status[task.status] += 1
            by_priority[task.priority] += 1
            if task.status != "done" and task.due_date:
                if validate_iso_date(task.due_date) < today:
                    overdue += 1

        done_count = by_status["done"]
        completion_pct = round((done_count / total) * 100, 1) if total else 0.0
        return {
            "total": total,
            "completion_pct": completion_pct,
            "overdue": overdue,
            "by_status": by_status,
            "by_priority": by_priority,
        }


def format_table(store: TaskStore, tasks: List[Task]) -> str:
    if not tasks:
        return "No matching tasks found."

    today = dt.date.today()
    rows: List[Tuple[str, str, str, str, str, str, str]] = []
    for t in tasks:
        unmet = store.unmet_dependencies(t)
        dep_str = ",".join(f"#{d}" for d in t.depends_on) if t.depends_on else "-"
        if unmet:
            dep_str += f" (wait:{','.join(f'#{u}' for u in unmet)})"

        due_str = t.due_date or "-"
        if t.status != "done" and t.due_date:
            d = validate_iso_date(t.due_date)
            if d < today:
                due_str += " [OVERDUE]"
            elif d == today:
                due_str += " [TODAY]"

        tags_str = ", ".join(t.tags) if t.tags else "-"
        score_str = str(store.urgency_score(t, today))
        rows.append(
            (
                f"#{t.id}",
                t.priority.upper(),
                score_str,
                t.status,
                due_str,
                dep_str,
                f"{t.title} [{tags_str}]" if tags_str != "-" else t.title,
            )
        )

    headers = ("ID", "PRIO", "SCORE", "STATUS", "DUE", "DEPS", "TASK")
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cols: Tuple[str, ...]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    sep = "  ".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="taskflow",
        description="CLI task & workflow engine with dependency & urgency tracking.",
    )
    parser.add_argument(
        "--store",
        default=str(DEFAULT_STORE),
        help=f"Path to JSON task store (default: {DEFAULT_STORE})",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Create a new task")
    p_add.add_argument("title", help="Task description/title")
    p_add.add_argument(
        "-p",
        "--priority",
        default="medium",
        choices=list(PRIORITY_WEIGHTS.keys()),
        help="Task priority level",
    )
    p_add.add_argument(
        "-t",
        "--tags",
        default="",
        help="Comma-separated tags (e.g., backend,api,v1)",
    )
    p_add.add_argument(
        "-d",
        "--depends-on",
        default="",
        help="Comma-separated prerequisite task IDs (e.g., 1,2)",
    )
    p_add.add_argument(
        "--due",
        default=None,
        help="Due date in YYYY-MM-DD format",
    )

    p_list = sub.add_parser("list", help="List and filter tasks")
    p_list.add_argument("--status", choices=list(VALID_STATUSES), default=None)
    p_list.add_argument(
        "--priority", choices=list(PRIORITY_WEIGHTS.keys()), default=None
    )
    p_list.add_argument("--tag", default=None)
    p_list.add_argument(
        "--ready",
        action="store_true",
        help="Show only incomplete tasks whose dependencies are all done",
    )
    p_list.add_argument(
        "--sort-urgency",
        action="store_true",
        help="Sort tasks by computed urgency score descending",
    )

    p_status = sub.add_parser("status", help="Update task status")
    p_status.add_argument("id", type=int, help="Task ID")
    p_status.add_argument("status", choices=list(VALID_STATUSES), help="New status")

    p_done = sub.add_parser("done", help="Mark a task as completed")
    p_done.add_argument("id", type=int, help="Task ID")

    p_del = sub.add_parser("delete", help="Delete a task")
    p_del.add_argument("id", type=int, help="Task ID")

    sub.add_parser("stats", help="Show completion and workload summary")
    return parser


def parse_int_csv(raw: str) -> List[int]:
    if not raw.strip():
        return []
    items: List[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            items.append(int(token))
        except ValueError as exc:
            raise TaskFlowError(f"Invalid integer ID '{token}'.") from exc
    return items


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        store = TaskStore(Path(args.store))

        if args.command == "add":
            tags = [t.strip() for t in args.tags.split(",") if t.strip()]
            deps = parse_int_csv(args.depends_on)
            task = store.add_task(
                title=args.title,
                priority=args.priority,
                tags=tags,
                depends_on=deps,
                due_date=args.due,
            )
            print(f"Added task #{task.id}: {task.title} [{task.priority}]")

        elif args.command == "list":
            tasks = store.filter_tasks(
                status=args.status,
                priority=args.priority,
                tag=args.tag,
                ready_only=args.ready,
            )
            if args.sort_urgency:
                today = dt.date.today()
                tasks.sort(
                    key=lambda t: (-store.urgency_score(t, today), t.id)
                )
            print(format_table(store, tasks))

        elif args.command == "status":
            task = store.set_status(args.id, args.status)
            print(f"Task #{task.id} status updated to '{task.status}'.")

        elif args.command == "done":
            task = store.set_status(args.id, "done")
            print(f"Completed task #{task.id}: {task.title}")

        elif args.command == "delete":
            store.delete_task(args.id)
            print(f"Deleted task #{args.id}.")

        elif args.command == "stats":
            stats = store.summary_stats()
            print(f"Total tasks : {stats['total']}")
            print(f"Completed   : {stats['completion_pct']}%")
            print(f"Overdue     : {stats['overdue']}")
            print("By status   : " + ", ".join(f"{k}={v}" for k, v in stats["by_status"].items()))
            print("By priority : " + ", ".join(f"{k}={v}" for k, v in stats["by_priority"].items()))

        return 0

    except TaskFlowError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
