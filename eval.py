from llava import ModelRunner
from teamcraft.utils.env_utils import NpEncoder, process_llava_output, find_closest_previous_time, concatenate_images_pure, extract_png_names, update_coordinates, filter_voxel
from teamcraft.utils.env_utils import get_initial_inp, get_middle_inp, get_initial_inp_dec, get_middle_inp_dec
from teamcraft.utils.demo_utils import TimeoutException, timeout_handler, process_json_files
# Utils
import argparse
import signal
import time
import json
import os
from PIL import Image
import numpy as np
from datetime import datetime

# Timeout
timeout_duration = 300  # seconds
signal.signal(signal.SIGALRM, timeout_handler)


class ConversationRecorder:
    """Class to record VLM conversations with image indices and timestamps"""
    
    def __init__(self, output_path, task_name, variant_id):
        self.output_path = output_path
        self.task_name = task_name
        self.variant_id = variant_id
        self.conversation_log = []
        self.image_frames = []
        self.step_counter = 0
        
        # Create directories for output
        self.conv_dir = os.path.join(output_path, 'conversations')
        self.gif_dir = os.path.join(output_path, 'gifs')
        os.makedirs(self.conv_dir, exist_ok=True)
        os.makedirs(self.gif_dir, exist_ok=True)
        
    def add_turn(self, role, content, images=None, reward=None, done=False):
        """Add a conversation turn with metadata"""
        turn = {
            'step': self.step_counter,
            'timestamp': datetime.now().isoformat(),
            'role': role,
            'content': content,
            'image_index': self.step_counter if images is not None else None,
            'reward': reward,
            'done': done
        }
        self.conversation_log.append(turn)
        
        # Store images for GIF creation
        if images is not None:
            # Convert images to PIL format if needed
            processed_images = []
            for img in images:
                if isinstance(img, np.ndarray):
                    processed_images.append(Image.fromarray(img))
                elif isinstance(img, Image.Image):
                    processed_images.append(img)
            
            # Create a composite image showing all perspectives
            if processed_images:
                # Calculate grid size
                n_images = len(processed_images)
                if n_images == 1:
                    composite = processed_images[0]
                else:
                    # Create a grid layout
                    cols = min(3, n_images)  # Max 3 columns
                    rows = (n_images + cols - 1) // cols
                    
                    # Get max dimensions
                    max_width = max(img.width for img in processed_images)
                    max_height = max(img.height for img in processed_images)
                    
                    # Create composite image
                    composite = Image.new('RGB', (cols * max_width, rows * max_height))
                    for idx, img in enumerate(processed_images):
                        row = idx // cols
                        col = idx % cols
                        composite.paste(img, (col * max_width, row * max_height))
                
                self.image_frames.append(composite)
        
        if role == "model":
            self.step_counter += 1
    
    def save_conversation(self):
        """Save conversation log as JSON"""
        filename = f"{self.task_name}_variant_{self.variant_id}_conversation.json"
        filepath = os.path.join(self.conv_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump({
                'task': self.task_name,
                'variant': self.variant_id,
                'total_steps': self.step_counter,
                'conversation': self.conversation_log
            }, f, indent=2)
        
        print(f"Saved conversation to {filepath}")
        return filepath
    
    def save_gif(self, fps=2):
        """Save image frames as GIF"""
        if not self.image_frames:
            print("No images to save as GIF")
            return None
        
        filename = f"{self.task_name}_variant_{self.variant_id}.gif"
        filepath = os.path.join(self.gif_dir, filename)
        
        # Save as GIF with specified frame rate
        self.image_frames[0].save(
            filepath,
            save_all=True,
            append_images=self.image_frames[1:],
            duration=1000//fps,  # Duration in milliseconds
            loop=0
        )
        
        print(f"Saved GIF with {len(self.image_frames)} frames to {filepath}")
        return filepath


def evaluate(mode, ckpt, tasks, var_low, var_high, mc_port, mineflayer_port, out_folder, load_4bit, load_8bit):
    llava = ModelRunner(model_path = ckpt, load_4bit = load_4bit, load_8bit = load_8bit)
    for env_name in tasks:
        if env_name=='build':
            from teamcraft import BuildEnv as Env
        elif env_name=='break':
            from teamcraft import BreakEnv as Env
        elif env_name=='farm':
            from teamcraft import FarmEnv as Env
        elif env_name=='smelt':
            from teamcraft import SmeltEnv as Env
        
        out_folder_task = out_folder + env_name
        env = Env(output_folder = out_folder_task, mc_port = mc_port, mineflayer_port = mineflayer_port)
        
        for t in range(var_low, var_high):
            # Initialize conversation recorder
            recorder = ConversationRecorder(out_folder_task, env_name, t)
            
            # Set alarm for time out
            signal.alarm(timeout_duration)
        
            try:
                # Reset environment
                reset_info = env.reset(t)
                # Get concatenated task image
                concatenate_images = concatenate_images_pure(reset_info[0])
                # Initial state
                images, state, inventory, done, reward = reset_info[1]
                
                # Record initial state
                recorder.add_turn(
                    role="system",
                    content=f"Task: {env_name}, Variant: {t}, Initial inventory: {inventory}",
                    images=[concatenate_images],
                    reward=reward,
                    done=done
                )
                
                if mode == "cen":
                    # Get initial prompt
                    inp_initial = get_initial_inp(env_name, inventory, env)
                    inp_list = []
                    inp_list.append(inp_initial)
                    
                    # Record initial prompt
                    recorder.add_turn(
                        role="user",
                        content=inp_initial
                    )
                    
                    # Action loop
                    for step_idx in range(env.action_length*2):
                        # First perspective images for all action bots
                        images_for_model = [images[key] for key in env.bot_list]
                        image_input = [concatenate_images]+images_for_model
                        
                        # Run model
                        llava_output = llava.run_once(inp_list, images=image_input)
                        actions = process_llava_output(llava_output, env.center_position)
                        
                        # Record model output with images
                        recorder.add_turn(
                            role="model",
                            content=llava_output.strip('<s>').strip('</s>'),
                            images=image_input,
                            reward=reward,
                            done=done
                        )
                        
                        # Step action
                        images, state, inventory, done, reward = env.step(actions)
                        
                        # Record history
                        inp_list.append(llava_output.strip('<s>').strip('</s>'))
                        new_inp = get_middle_inp(env_name, inventory, env)
                        inp_list.append(new_inp)
                        
                        # Record user follow-up
                        recorder.add_turn(
                            role="user",
                            content=new_inp,
                            reward=reward,
                            done=done
                        )
                        
                        print(f"Step {step_idx}: Current reward for task {env_name}, variant {t} is {reward}", "task is done" if done else "")
                        
                        if done:
                            break

                else:  # Decentralized mode
                    # Get initial prompt for each bot
                    inp_list = []
                    for a in env.bot_list:
                        inp_initial = get_initial_inp_dec(env_name, inventory, env, a)
                        inp_list.append([inp_initial])
                        
                        # Record initial prompt for each bot
                        recorder.add_turn(
                            role="user",
                            content=f"Bot {a}: {inp_initial}"
                        )
                        
                    # Action loop
                    for step_idx in range(env.action_length*2):
                        actions = []
                        all_step_images = []
                        all_step_outputs = []
                        
                        for i_a, a in enumerate(env.bot_list):
                            # First perspective images for current action bot
                            images_a = [images[a]]
                            image_input = [concatenate_images] + images_a
                            all_step_images.extend(image_input)
                            
                            # Run model
                            llava_output = llava.run_once(inp_list[i_a], images=image_input)
                            actions += process_llava_output(llava_output, env.center_position)
                            all_step_outputs.append(f"Bot {a}: {llava_output.strip('<s>').strip('</s>')}")
                            
                            # Record history
                            inp_list[i_a].append(llava_output.strip('<s>').strip('</s>'))
                            new_inp = get_middle_inp_dec(env_name, inventory, env, a, i_a)
                            inp_list[i_a].append(new_inp)
                        
                        # Record all bot outputs for this step
                        recorder.add_turn(
                            role="model",
                            content="\n".join(all_step_outputs),
                            images=all_step_images[:4],  # Limit to first 4 images for GIF
                            reward=reward,
                            done=done
                        )
                        
                        # Step actions for all action bots
                        images, state, inventory, done, reward = env.step(actions)
                        
                        print(f"Step {step_idx}: Current reward for task {env_name}, variant {t} is {reward}", "task is done" if done else "")
                        
                        if done:
                            break
                
                # Save conversation and GIF
                recorder.save_conversation()
                recorder.save_gif(fps=2)  # 2 frames per second
                
                print(f"Task {env_name} variant {t} completed with no issue.")
                print("----------------------------------------------------------------------------")
                
            except TimeoutException:
                print(f"Task {env_name} variant {t} timed out in {timeout_duration} seconds.")
                # Still save what we have
                recorder.save_conversation()
                recorder.save_gif(fps=2)
                print("----------------------------------------------------------------------------")
                
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                # Still save what we have
                recorder.save_conversation()
                recorder.save_gif(fps=2)
                env.env = None
                time.sleep(10)
                print("----------------------------------------------------------------------------")
                
            finally:
                # Disable alarm after code completes or fails
                signal.alarm(0)

        env.close()
        folder_path = out_folder_task + '/json/'
        eval_set = {'build':['test','shape','material','scene','agents'],
                    'break':['test','shape','material','scene','agents'],
                    'farm':['test','none','crop','scene','agents'],
                    'smelt':['test','goal','furnace','scene','agents'],
                    }
        
        # aggregate results for full evaluation of a task
        if var_high == 250 and var_low == 0:
            for i_t in range(5):
                start_index = i_t * 50
                end_index = (i_t + 1) * 50 - 1
                average_reward, done_true_percentage = process_json_files(folder_path, start_index, end_index)
                print("average reward and success percentage of " + out_folder_task + ' ' + eval_set[env_name][i_t], average_reward, done_true_percentage)
        
        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to evaluate a checkpoint with conversation and GIF recording")

    parser.add_argument("--mode", type=str, default="cen", help="Evaluation mode, could be cen (centralized) or dec (decentralized)")
    parser.add_argument("--ckpt", type=str, default="teamcraft/TeamCraft-VLA-7B-Cen", help="Local path or huggingface name of the checkpoint")
    parser.add_argument("--mc_port", type=int, default=2037, help="Minecraft server port")
    parser.add_argument("--tasks", nargs='+', type=str, default=['build','break','farm','smelt'], help="Task names, could be build, break, farm, or smelt")
    parser.add_argument("--var_low", type=int, default=0, help="Lowest task variant seed, could be 0 to 249")
    parser.add_argument("--var_high", type=int, default=1, help="One above the largest task variant seed, could be 1 to 250")
    parser.add_argument("--mineflayer_port", type=int, default=3000, help="Mineflayer server port")
    parser.add_argument("--out_folder", type=str, default="eval_cen/", help="Output folder path")
    parser.add_argument("--load_4bit", type=bool, default=False, help="Whether to load the checkpoint with 4 bit")
    parser.add_argument("--load_8bit", type=bool, default=False, help="Whether to load the checkpoint with 8 bit")

    args = parser.parse_args()
    assert args.var_high > args.var_low, "var_high should be greater than var_low"
    assert args.mode in ["cen", "dec"], "mode should be either cen or dec"
    print("Evaluating the tasks: %s" % args.tasks)
    print("Evaluating the variants: [%s, %s)" % (args.var_low, args.var_high))
    evaluate(args.mode, args.ckpt, args.tasks, args.var_low, args.var_high, args.mc_port, args.mineflayer_port, args.out_folder, args.load_4bit, args.load_8bit)
