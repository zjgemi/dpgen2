from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Sequence,
)
from typing import (
    Dict,
    List,
    Tuple,
)

from .task import (
    ExplorationTask,
)


class ExplorationTaskGroup(Sequence):
    """A group of exploration tasks. Implemented as a `list` of `ExplorationTask`."""

    def __init__(self):
        super().__init__()
        self.clear()

    def __getitem__(self, ii: int) -> ExplorationTask:
        """Get the `ii`th task"""
        return self.task_list[ii]

    def __len__(self) -> int:
        """Get the number of tasks in the group"""
        return len(self.task_list)

    def clear(self) -> None:
        self._task_list = []

    @property
    def task_list(self) -> List[ExplorationTask]:
        """Get the `list` of `ExplorationTask`"""
        return self._task_list

    def add_task(self, task: ExplorationTask):
        """Add one task to the group."""
        self.task_list.append(task)
        return self

    def add_group(
        self,
        group: "ExplorationTaskGroup",
    ):
        """Add another group to the group."""
        # see https://www.python.org/dev/peps/pep-0484/#forward-references for forward references
        self._task_list = self._task_list + group._task_list
        return self

    def __add__(
        self,
        group: "ExplorationTaskGroup",
    ):
        """Add another group to the group."""
        return self.add_group(group)

    @abstractmethod
    def make_task(self) -> "ExplorationTaskGroup":
        """Make the task group."""
        pass


class ExplorationTaskGroupData(ExplorationTaskGroup):
    """Data-only exploration task group."""

    def __init__(self):
        super().__init__()

    def make_task(self):
        """Make the task group."""
        raise NotImplementedError("This class is not supposed to supply make_task")


class FooTaskGroup(ExplorationTaskGroup):
    def __init__(self, numb_task):
        super().__init__()
        # TODO: confirm the following is correct
        self.tlist = ExplorationTaskGroup()
        for ii in range(numb_task):
            self.tlist.add_task(
                FooTask(
                    f"conf.{ii}",
                    f"this is conf.{ii}",
                    f"input.{ii}",
                    f"this is input.{ii}",
                )
            )

    @property
    def task_list(self):
        return self.tlist


if __name__ == "__main__":
    grp = FooTaskGroup(3)
    for ii in grp:
        fcs = ii.files()
        print(fcs)
