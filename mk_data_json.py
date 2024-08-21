import numpy as np
import pandas as pd
import os
import glob
import json
from tqdm import tqdm
import argparse

def load_file_lists(data_path):
    file_patterns = {
        'ct': 'CT/*.nii.gz',
        'ct_norm': 'CT_norm/*.nii.gz',
        'ct_norm_lung': 'CT_norm_lung/*.nii.gz',
        'ct_norm_lung1024': 'CT_norm_lung1024/*.nii.gz',
        'ct_label': 'CT_label/*.nii.gz',
        'mr': 'MR/*.nii.gz',
        'mr_norm_slice': 'MR_norm_slice/*.nii.gz',
        'mr_norm_img': 'MR_norm_img/*.nii.gz',
        'mr_edge': 'MR_CannyEdgeMap/*.png',
        'mr_label': 'MR_label/*.nii.gz'
    }

    return {key: sorted(glob.glob(os.path.join(data_path, pattern))) for key, pattern in file_patterns.items()}

def create_data_dict(i, file_lists, prompt, mr_label_list_for_check):
    ct_path = file_lists['ct'][i]
    mr_path = file_lists['mr'][i]

    # Check if filenames match between MR and CT (ignoring prefixes)
    if not os.path.basename(ct_path).replace('MR', '').replace('CT', '') == os.path.basename(mr_path).replace('MR', '').replace('CT', ''):
        print('[WARNING] Mismatch found')
        print(f'CT: {ct_path} \nMR: {mr_path}')

    # Create the dictionary with relative paths
    temp_dict = {
        'source_norm_slice': os.path.join('MR_norm_slice', os.path.basename(file_lists['mr_norm_slice'][i])),
        'source_norm_img': os.path.join('MR_norm_img', os.path.basename(file_lists['mr_norm_img'][i])),
        'source_edge': os.path.join('MR_CannyEdgeMap', os.path.basename(file_lists['mr_edge'][i])),
        'target_norm': os.path.join('CT_norm', os.path.basename(file_lists['ct_norm'][i])),
        'target_norm_le1': os.path.join('CT_norm_lung1024', os.path.basename(file_lists['ct_norm_lung1024'][i])),
        'target_norm_le2': os.path.join('CT_norm_lung', os.path.basename(file_lists['ct_norm_lung'][i])),
        'prompt': prompt,
        'source_ori': os.path.join('MR', os.path.basename(mr_path)),
        'target_ori': os.path.join('CT', os.path.basename(ct_path)),
        'target_label': os.path.join('CT_label', os.path.basename(file_lists['ct_label'][i]))
    }

    # Check if MR label exists for this entry
    temp = os.path.basename(file_lists['ct_norm'][i]).replace('CT_norm', '')
    temp_dict['MR_label'] = ''
    if temp in mr_label_list_for_check:
        idx2 = mr_label_list_for_check.index(temp)
        mr_label_path = file_lists['mr_label'][idx2]
        temp_dict['MR_label'] = os.path.join('MR_label', os.path.basename(mr_label_path))

    return temp_dict

def parse_subject_list(subjects_str):
    """
    Convert a comma-separated string into a list of subjects.
    """
    return subjects_str.split(',') if subjects_str else []

def main(data_path, file_name, val_list_str, test_list_str):
    val_list = parse_subject_list(val_list_str)
    test_list = parse_subject_list(test_list_str)

    train_dict = []
    val_dict = []
    test_dict = []

    # Load file lists
    file_lists = load_file_lists(data_path)

    # Extract just the filenames from the MR labels for checking
    mr_label_list_for_check = [os.path.basename(l).replace('lung_mr', '') for l in file_lists['mr_label']]

    prompt = 'Professional high-quality translation from lung MR to CT. Magnetic Resonance Imaging to Computed Tomography, Medical Imaging, extremely high detail, Clean Background,'

    for i in range(len(file_lists['ct'])):
        temp_dict = create_data_dict(i, file_lists, prompt, mr_label_list_for_check)
        patient = os.path.basename(file_lists['ct'][i])[:10]

        # Append to the appropriate list
        if patient in test_list:
            test_dict.append(temp_dict)
        elif patient in val_list:
            val_dict.append(temp_dict)
        else:
            train_dict.append(temp_dict)

    print(f'Training length: {len(train_dict)}, Validation length: {len(val_dict)}, Test length: {len(test_dict)} => Total: {len(train_dict) + len(val_dict) + len(test_dict)}')

    final_dict = {
        'description':'MR2CT for lung segmentation 2D',
        'numTest':len(test_dict),
        'numValidation':len(val_dict),
        'numTrain':len(train_dict),
        'training':train_dict,
        'validation':val_dict,
        'test':test_dict
    }

    with open(os.path.join(data_path, file_name), "w") as f:
        json.dump(final_dict, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate JSON for MR2CT for Lung segmentation")
    parser.add_argument('--data_path', type=str, required=True, help='Root path of the dataset')
    parser.add_argument('--file_name', type=str, required=True, help='Name of the JSON file')
    parser.add_argument('--val_list', type=str, default='', help='Comma-separated list of subjects for validation')
    parser.add_argument('--test_list', type=str, default='', help='Comma-separated list of subjects for testing')
    
    args = parser.parse_args()
    main(args.data_path, args.file_name, args.val_list, args.test_list)