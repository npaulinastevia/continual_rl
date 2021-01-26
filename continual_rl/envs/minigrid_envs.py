import numpy as np
from gym_minigrid.envs.dynamicobstacles import DynamicObstaclesEnv
from gym_minigrid.envs import MiniGridEnv, WorldObj, Grid, Goal, COLORS, Floor
from gym_minigrid.rendering import fill_coords, point_in_rect


class DynamicObstaclesRandomEnv8x8(DynamicObstaclesEnv):
    def __init__(self):
        """
        See: https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/envs/dynamicobstacles.py
        """
        super().__init__(size=8, n_obstacles=4, agent_start_pos=None)


class FakeGoal(WorldObj):
    def __init__(self, color='green'):
        super().__init__('lava', color)

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class SimpleChoiceEnv(MiniGridEnv):
    """
    Based on the EmptyEnv, only instead of one fixed-place goal, the green block is now negative, and the true
    goal is the red goal next to it.
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a green lava (looks like a goal, but it's a trap)
        lava = FakeGoal()
        self.put_obj(lava, width - 2, height - 2)

        # Place a goal square in the bottom-right corner. The goal is red, because we're being tricky
        goal = Goal()
        goal.color = 'red'
        self.put_obj(goal, width - 4, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the red goal square"


class OddManOutEnv(MiniGridEnv):
    """
    In this env there are 3 possible goal locations. The one that is the odd color out is the one that, when selected,
    gives reward. The other 2 end the episode.
    """
    def __init__(
        self,
        correct_color, incorrect_color, num_choices=3,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self._num_choices = num_choices
        self._correct_color = correct_color
        self._incorrect_color = incorrect_color

        # Red is the "default" color - 0, so empty/unseen seems to come out "red" when type masked
        # Grey is wall color
        assert 'red' not in self._correct_color and 'red' not in self._correct_color
        assert 'grey' not in self._correct_color and 'grey' not in self._correct_color

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_box_order(self):
        # n-1 incorrect options (represented with False), 1 correct option (represented with True)
        box_order = [False for _ in range(self._num_choices-1)]
        box_order.append(True)

        # Shuffles in-place
        random_state = np.random.RandomState()
        random_state.shuffle(box_order)
        return box_order

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate the random boxes and place them
        boxes = self._gen_box_order()

        for box_id, box_correct in enumerate(boxes):
            if box_correct:
                box = Goal()
                box.color = self._correct_color
            else:
                # Place a green lava (looks like a goal, but it's a trap)
                box = FakeGoal(color=self._incorrect_color)

            box_x = box_id % (width//2 - 1)  # So they're spaced 2 apart, and don't include walls
            box_y = box_id // (height//2 - 1)

            self.put_obj(box, width - 2 * (box_x + 1), height - 2 * (box_y + 1))

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = f"get to the {self._correct_color} goal square"

    def gen_obs(self):
        """
        Override the original gen_obs to strip out the object_index. Just masking it out so it can still be
        used alongside environments that still use object index. The reason we do this is because it ruins our
        association game if the agent can see which box is lava.
        """
        # TODO: is this really what I want to do? vs cutting it out?
        obs = super().gen_obs()
        obs['image'][:, :, 0] = 0
        obs['image'] *= 2  # The signal is pretty small now, so boost it a little (still consistent with 10 as max)

        return obs


class AssociationEnv(MiniGridEnv):
    """
    This test is meant to mimic the classic catastrophic forgetting test (from...McCloskey? Ratclif? TODO: check) where
    the agent is trained on A->B associations, then on A-C associations. (I.e. the agent is shown a word from list A,
    and is expected to provide the appropriate B answer during the first task, then a different C answer in the second.)
    For this, the associations are color-color mappings.

    To indicate if we're in the first task or the second, there will be an indicator box that does nothing other than
    be the task's color.
    To indicate what "A" is there will be another box with the selected A's color.
    To provide options for B, there will be a set of n boxes with the options
    """
    def __init__(
        self,
        association_pairs,
        indicator_color,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self._association_pairs = association_pairs
        self._indicator_color = indicator_color

        # Make sure there are no duplicate colors in our associations
        association_a = [association_pair[0] for association_pair in association_pairs]
        association_b = [association_pair[1] for association_pair in association_pairs]
        assert(len(set(association_a))) == len(association_pairs)
        assert(len(set(association_b))) == len(association_pairs)
        assert 'grey' not in association_a and 'grey' not in association_b, "Indistinguishable from environment"
        assert 'red' not in association_a and 'red' not in association_b, "Indistinguisable from environment"

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _select_association(self):
        import copy
        box_order = copy.deepcopy(self._association_pairs)  # Just so we don't change the original at all

        # Shuffles in-place
        random_state = np.random.RandomState()
        random_state.shuffle(box_order)

        right_answer = box_order[random_state.randint(len(box_order))]

        return box_order, right_answer

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate the random boxes and place the possible answers
        box_options, right_answer = self._select_association()

        # Place the possible answers
        for box_id, association_opt in enumerate(box_options):
            (clue_color, answer_color) = association_opt

            if clue_color == right_answer[0]:
                assert answer_color == right_answer[1]
                box = Goal()
                box.color = answer_color
            else:
                # If this is an answer to a question that isn't the one we're asking (aka not the selected clue)
                box = FakeGoal(color=answer_color)

            box_x = box_id % (width//2 - 1)  # So they're spaced 2 apart, and don't include walls
            box_y = box_id // (height//2 - 1)

            self.put_obj(box, width - 2 * (box_x + 1), height - 2 * (box_y + 2))

        # Place the clue and the indicator
        indicator = Floor(color=self._indicator_color)
        self.put_obj(indicator, width - 2, height - 2)
        clue = Floor(color=right_answer[0])
        self.put_obj(clue, width - 4, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = f"get to the square that is associated with the given clue"

    def gen_obs(self):
        """
        Override the original gen_obs to strip out the object_index. Just masking it out so it can still be
        used alongside environments that still use object index. The reason we do this is because it ruins our
        association game if the agent can see which box is lava.
        """
        # TODO: is this really what I want to do? vs cutting it out?
        obs = super().gen_obs()
        obs['image'][:, :, 0] = 0
        obs['image'] *= 2  # The signal is pretty small now, so boost it a little (still consistent with 10 as max)

        return obs
