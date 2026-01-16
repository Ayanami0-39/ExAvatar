import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Generate frame lists.")
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the output files')
    
    # Arguments for frame_list_all: start, end
    parser.add_argument('--all', type=int, nargs=2, metavar=('START', 'END'), required=True, 
                        help='Start and end frame for frame_list_all (inclusive)')
    
    # Arguments for frame_list_train: start, step
    parser.add_argument('--train', type=int, nargs=2, metavar=('START', 'STEP'), required=True, 
                        help='Start frame and step for frame_list_train')
    
    # Arguments for frame_list_test: start, step
    parser.add_argument('--test', type=int, nargs=2, metavar=('START', 'STEP'), required=True, 
                        help='Start frame and step for frame_list_test')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Parse 'all' arguments
    all_start, all_end = args.all
    if all_start > all_end:
        print(f"Error: all_start ({all_start}) > all_end ({all_end})")
        return

    # Generate frame_list_all
    all_frames = list(range(all_start, all_end + 1))
    all_file = os.path.join(args.output_dir, 'frame_list_all.txt')
    with open(all_file, 'w') as f:
        for frame in all_frames:
            f.write(f"{frame}\n")
    print(f"Generated {all_file} with {len(all_frames)} frames.")

    max_frame = all_end

    # Parse 'train' arguments
    train_start, train_step = args.train
    train_frames = []
    # Generate frame_list_train
    # Ensure we don't go beyond max_frame
    for frame in range(train_start, max_frame + 1, train_step):
        train_frames.append(frame)
        
    train_file = os.path.join(args.output_dir, 'frame_list_train.txt')
    with open(train_file, 'w') as f:
        for frame in train_frames:
            f.write(f"{frame}\n")
    print(f"Generated {train_file} with {len(train_frames)} frames.")

    # Parse 'test' arguments
    test_start, test_step = args.test
    test_frames = []
    # Generate frame_list_test
    for frame in range(test_start, max_frame + 1, test_step):
        test_frames.append(frame)
        
    test_file = os.path.join(args.output_dir, 'frame_list_test.txt')
    with open(test_file, 'w') as f:
        for frame in test_frames:
            f.write(f"{frame}\n")
    print(f"Generated {test_file} with {len(test_frames)} frames.")

if __name__ == '__main__':
    main()
