#!/usr/bin/env python3

#####################################################################
# This script presents how to use the most basic features of the environment.
# It configures the engine, and makes the agent perform random actions.
# It also gets current state and reward earned with the action.
# <episodes> number of episodes are played.
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
#
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

import os
import pickle
from random import choice

import vizdoom as vzd
from PIL import Image

# class DoomData(vzd.DoomGame):
#     def complete_action(self, choice, tics):
#         self.make_action(choice)
#         self.make_action(
#             [False, False, False], tics - 1
#         )  # change based on num of action


# Create DoomGame instance. It will run the game and communicate with you.
game = vzd.DoomGame()

# Now it's time for configuration!
# load_config could be used to load configuration instead of doing it here with code.
# If load_config is used in-code configuration will also work - most recent changes will add to previous ones.
# game.load_config("../../scenarios/basic.cfg")

# Sets path to additional resources wad file which is basically your scenario wad.
# If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
game.set_doom_game_path("./doom.wad")
game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "basic.wad"))
game.set_doom_map("map01")
game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
# 24 because 3 * 8 -> 2^8 = 255. aka standard RGB scheme.
game.set_screen_format(vzd.ScreenFormat.RGB24)
game.set_depth_buffer_enabled(False)
game.set_labels_buffer_enabled(False)
game.set_automap_buffer_enabled(False)
game.set_objects_info_enabled(False)
game.set_sectors_info_enabled(False)

game.set_render_hud(False)
game.set_render_minimal_hud(False)  # If hud is enabled
game.set_render_crosshair(False)
game.set_render_weapon(True)
game.set_render_decals(False)  # Bullet holes and blood on the walls
game.set_render_particles(False)
game.set_render_effects_sprites(False)  # Like smoke and blood
game.set_render_messages(False)  # In-game text messages
game.set_render_corpses(False)
game.set_render_screen_flashes(False)  # Effect upon taking damage or picking up items

# Adds buttons that will be allowed to use.
# This can be done by adding buttons one by one:
# game.clear_available_buttons()
# game.add_available_button(vzd.Button.MOVE_LEFT)
# game.add_available_button(vzd.Button.MOVE_RIGHT)
# game.add_available_button(vzd.Button.ATTACK)
# Or by setting them all at once:
game.set_available_buttons(
    [vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK]
)
# Buttons that will be used can be also checked by:
print("Available buttons:", [b.name for b in game.get_available_buttons()])

# Adds game variables that will be included in state.
# Similarly to buttons, they can be added one by one:
# game.clear_available_game_variables()
# game.add_available_game_variable(vzd.GameVariable.AMMO2)
# Or:
game.set_available_game_variables([vzd.GameVariable.AMMO2])
print(
    "Available game variables:",
    [v.name for v in game.get_available_game_variables()],
)

# Causes episodes to finish after 200 tics (actions)
game.set_episode_timeout(805)

# Makes episodes start after 10 tics (~after raising the weapon)
game.set_episode_start_time(10)

# Makes the window appear (turned on by default)
# game.set_window_visible(True)
game.set_window_visible(False)  # ! CHANGE -- FOR TESTING PURPOSES
game.set_living_reward(-1)
game.set_mode(vzd.Mode.PLAYER)

# Initialize the game. Further configuration won't take any effect from now on.
game.init()
actions = [
    [True, False, False],
    [False, True, False],
    [False, False, True],
    [False, False, False],
]

# Run this many episodes
# Sets time that will pause the engine after each action (in seconds)
# Without this everything would go too fast for you to keep track of what's happening.

total_frames = 0
i = 1
alr_done = False

while total_frames < 160_000:
    print(f"Episode #{i}")

    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game.new_episode()
    ep_len = 0
    # Loading animation prevents any other action from going through
    game.make_action([False, False, False], 5)

    action_set = []
    image_memory = []
    if not alr_done:
        os.mkdir(f"./frames_dataset_test/episode_{i}")
    while not game.is_episode_finished():
        ep_len += 1
        # Gets the state
        state = game.get_state()
        screen_buf = state.screen_buffer
        image_memory.append(screen_buf)

        a = choice(actions)

        if a == [False, False, True] and 2 in action_set[-13:]:
            a = [False, False, False]
        # r = game.complete_action(a, 7)
        r = game.make_action(a)
        action_set.append(a.index(True) if True in a else 3)

    vals_to_remove = ep_len % 16
    image_memory = image_memory[:-vals_to_remove]
    action_set = action_set[:-vals_to_remove]

    if len(image_memory) > 0:
        with open(f"./frames_dataset_test/episode_{i}/actions.pkl", "wb") as f:
            pickle.dump(action_set, f)

        for idx, img in enumerate(image_memory):    
            Image.fromarray(img).save(f"./frames_dataset_test/episode_{i}/{idx + 1}.png")

        total_frames += ep_len - vals_to_remove
        i += 1

        alr_done = False
    else:
        alr_done = True
# It will be done automatically anyway but sometimes you need to do it in the middle of the program...
game.close()
