""" Scripts to download datasets and checkpoints. """

import os
import requests
from tqdm import tqdm
from pathlib import Path
import subprocess
from urllib.parse import urlsplit
import zipfile
import tarfile
from typing import Optional, List

# URL of TUM RGB-D dataset
URLs = dict(
    TUM_RGBD = "https://vision.in.tum.de/rgbd/dataset/",
    Bonn_RGBD = "https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/",
)

TUM_RGBD_Sequences = {
    "freiburg1": [
        "rgbd_dataset_freiburg1_360",
        "rgbd_dataset_freiburg1_desk",
        "rgbd_dataset_freiburg1_desk2",
        "rgbd_dataset_freiburg1_floor",
        "rgbd_dataset_freiburg1_plant",
        "rgbd_dataset_freiburg1_room",
        "rgbd_dataset_freiburg1_rpy",
        "rgbd_dataset_freiburg1_xyz",
    ],
    "freiburg2": [
        "rgbd_dataset_freiburg2_360_hemisphere",
        "rgbd_dataset_freiburg2_360_kidnap",
        "rgbd_dataset_freiburg2_desk",
        "rgbd_dataset_freiburg2_desk_with_person",
        "rgbd_dataset_freiburg2_dishes",
        "rgbd_dataset_freiburg2_large_no_loop",
        "rgbd_dataset_freiburg2_large_with_loop",
        "rgbd_dataset_freiburg2_pioneer_360",
        "rgbd_dataset_freiburg2_pioneer_slam",
        "rgbd_dataset_freiburg2_pioneer_slam2",
        "rgbd_dataset_freiburg2_pioneer_slam3",
        "rgbd_dataset_freiburg2_rpy",
        "rgbd_dataset_freiburg2_xyz",
    ],
    "freiburg3": [
        "rgbd_dataset_freiburg3_long_office_household",
        # "rgbd_dataset_freiburg3_nostructure_notexture_far",
        # "rgbd_dataset_freiburg3_nostructure_notexture_near",
        # "rgbd_dataset_freiburg3_nostructure_texture_far",
        # "rgbd_dataset_freiburg3_nostructure_texture_near",
        "rgbd_dataset_freiburg3_sitting_halfsphere",
        "rgbd_dataset_freiburg3_sitting_rpy",
        "rgbd_dataset_freiburg3_sitting_static",
        "rgbd_dataset_freiburg3_sitting_xyz",
        # "rgbd_dataset_freiburg3_structure_notexture_far",
        # "rgbd_dataset_freiburg3_structure_notexture_near",
        # "rgbd_dataset_freiburg3_structure_texture_far",
        # "rgbd_dataset_freiburg3_structure_texture_near",
        "rgbd_dataset_freiburg3_teddy",
        "rgbd_dataset_freiburg3_walking_halfsphere",
        "rgbd_dataset_freiburg3_walking_rpy",
        "rgbd_dataset_freiburg3_walking_static",
        "rgbd_dataset_freiburg3_walking_xyz",
    ],
}

Bonn_RGBD_Sequences = {
    "seq": [
        "rgbd_bonn_crowd",
        "rgbd_bonn_kidnapping_box2",
        "rgbd_bonn_moving_nonobstructing_box",
        "rgbd_bonn_placing_obstructing_box",
        "rgbd_bonn_static",
        "rgbd_bonn_balloon",
        "rgbd_bonn_placing_nonobstructing_box3",
        "rgbd_bonn_moving_nonobstructing_box2",
        "rgbd_bonn_balloon_tracking",
        "rgbd_bonn_moving_obstructing_box",
        "rgbd_bonn_person_tracking2",
        "rgbd_bonn_removing_nonobstructing_box",
        "rgbd_bonn_placing_nonobstructing_box2",
        "rgbd_bonn_synchronous",
        "rgbd_bonn_crowd3",
        "rgbd_bonn_removing_obstructing_box",
        "rgbd_bonn_placing_nonobstructing_box",
        "rgbd_bonn_balloon2",
        "rgbd_bonn_crowd2",
        "rgbd_bonn_synchronous2",
        "rgbd_bonn_removing_nonobstructing_box2",
        "rgbd_bonn_static_close_far",
        "rgbd_bonn_moving_obstructing_box2",
        "rgbd_bonn_kidnapping_box",
        "rgbd_bonn_person_tracking",
        "rgbd_bonn_balloon_tracking2",
    ],
}

Oxford_Multimotion_sequences = {
    'primary': {       
        # sequence_name: [rgbd.tgz, imu.csv, vicon.csv]
        'cars_3_unconstrained': [
            '15CJ_0xelTYDNtSZJEczsw2ZOSmMKaITg', '1jnR1ShFklEq7MR-c6whbbKbb1LYHYsnQ', '1qlsr1S1OF6s4pclc5pe3Zkgqjp0GtQKH'
        ],
        'cars_3_translational': [
            '1fuKJzsarGv1BhdZgfGJzyIYd38G9HIuN', '14Y2moyU_FHBp8xa0uSeNpiEcwA00HPtK', '1xCKeV5WLpBhfi6cL1DTkAFS8UkD_NPI2'
        ],
        'cars_3_static': [
            '1AmnpfTmPtUL7obkPsf4vLWfiDlTqoRsQ', '1RvvWEDynH5c1n5gJD13xIMZGDPuSJOw2', '1GO3o_cd2yofFSylBE8yXestFOjmZ32W0'
        ],
        'swinging_4_unconstrained': [
            '1zr25KBRuErEB_jYzs6xMRQ_RtuN2I_uD', '1RQ2dQV_QTV_YC6T9HPaWjLAJia09kqZC', '1cqPqTMYuvRoNM86gs8-LjaO3mE6mP0Me'
        ],
        'swinging_4_translational': [
            '1UHcO_KivhWqKH9dL1Z5iFObXYYNQOrrw', '1AU5twQtslNAbEhbov_UNcAwcRiZJStdO', '1iW1C1DRYiK1ozlXpIHL8y6Rclw-M5bH3'
        ],
        'swinging_4_static': [
            '1osIGWMPwSC0N-8hMQAzxJ2aTxRo5Gm__', '15vXt3jedX9qjLjByOw9WW765KFdpkFHo', '1rogJSvTM9W53DQmnW3hA4rv4ZNsmF7HX'
        ] ,
        'cars_6_unconstrained': [
            '13DOl7x-1zJxDI9AKEiJ3FxE4vnMf1dJw', '1F47uEc-476EvtMlb9IFGjmu7S-SUGlrx', '1AY--RFyVjxWyH5702qBQuVI28QXbOfMX'
        ],
        'cars_6_translational': [
            '1Dpv5GSgBEyS-27O6-ecVfwlUCsTCGXSL', '1vtUZxlaEAzEhbV_LIWUFWWjJ__9uLAjd', '1WQ-HZYXxfDOoAtn7xmES8HFauK769Id7'
        ],
        'cars_6_static': [
            '1k2gGreplsMS_v3eZD6Hjx6LuK4TmOujQ', '1RVaaA_8T59rnbFuNgsZJnx5ekzgcBl9x', '1uYqvi3Lr5-0c4WrVA11dCvNSFJTQEfFW'
        ],
        'cars_6_robot': [
            '1POiW1UxuxQSXwLeQl9a89b-OXTVENs1_', '1_vfzAIK4ZMolfy55CP85CQ4Z_MjCA8Mi', '1dGWyMQylq2t0OrypFYuBd-teVraTK0lf'
        ],
        'occlusion_2_unconstrained': [
            '1t2vyw-H18NwBoVFbtJ3AT7eBl3uucamg', '1QWcu792DqqDFIl8WB8mDeGo6q6XsXk9o', '1C7_FBEAqgUmLJCuZdAXG5DzSvsYdmj0j'
        ],
        'occlusion_2_translational': [
            '1T74pIVQ4DWVp9veZzoHFsvZyYPIr1Yck', '1l-TikfRAPzowwF3pFMrOa2BjJeu_JUqD', '1ABy3HMlz5Wm1GFnKtmm4Hr9W9a03ua-F'
        ],
        'occlusion_2_static': [
            '1ntklaWH3VOlnkxG3k2D8vKiq7GgWbQPM', '1Ly_SkLeydeYBCSPduokAKL_PkD-LMqkl', '142Rsb8T8IyWr0dTKiAAeMWLkTmODg94T'
        ],
    },
    'secondary': {
        'calibration_extrinsic_no_imu': [ 
            '1CW9p4ku1IWqbISLOJd4OnqsG5XqItdYn', '1uPBEWFGbQqLFc5d2kWu7hiOnNjIdt3Nk'
        ],
        'calibration_extrinsic': [
            '1CpNiqrX8Jqcjxtmy-2SzcD7XwYTG7N5B', '1xFY-E-F0o3_oXqgfrkwKWOVpU62e0pHm', '1D379VPgxy3TFphhUiqyBQbVUWFsCPR1x'
        ],
        'calibration_vicon_no_imu': [
            '1_cJOBulEHDFHZ4IdiubphTDBlmEeVTLn', '1zCL9B27KVMw1vTZ34fkc9l1js7o7P_Hl'
        ],
        'pinwheel_1_unconstrained': [
            '1A1JfKDAsUkmZAvn_yE-fcc0NybDMFFX7', '1gFhwfZEKR893Ir2PaJbD2HnVsXHObpeO', '1aN7DDn1eqmUQo2SNMuP7Obm-bxKdcWBP'
        ],
        'pinwheel_1_static': [
            '1rQbZ4MfjYNpNDTE7HT0Xk3x72CzS1yOz', '1Kpyh1Ney8sowXGGpcGvN5FvycaU_yTOy', '15ctKENVEImPzEbnjU6d5hHrrKLaWLHbg'
        ],
        'fixed_occlusion_1_unconstrained': [
            '1yY3ZqxUEEVEZoWdebes-pDKIr8tgEnCL', '1CaQURsZPTpA_CbUithFxfjlfHKfQbdvv', '16xU_ComEKtbiccucZYj8Q8_wjY44_tDH'
        ],
        'fixed_occlusion_1_static': [
            '1T3u2ZH0WmL1mba9uQvmC6ExpABVfkm7P', '1OvL9sOIfqYQuqV78gca04rtuxCkdyY_f', '1W7R42jGIyIpt1rb_t9fiQA8fJskmzXmQ'
        ],
        'fixed_occlusion_1_translational': [
            '1rnaMIDwc9B37_Ukza2-2TO9zM0qWf-Nn', '1WQ2-Bzu2He8zVwO23QNQV156yESPd-ds', '19iy74JtolhwEKNLEZz5Okk5hvVgnfhyW'
        ],
        'cars_1_unconstrained': [
            '1oJngwPyeETpYCA_CHpBD2GXkXggPD5wE', '1IhHBhXX8BiISMIcOOKUAuNW_NzFMzV5f', '1GOxXFS7slmRxUyP6AYBUFyBNDlir1OU5'
        ],
        'calibration_vicon': [
            '1XWPizQj044jtLN3ccrneSIBJFRiIvTpj', '1_T1e2hGiofwqVtsW4KlPEjZ4qv8RWwIG', '1mkF_0aSuiJn7lMfam_KrW-xs9yB1hNXt'
        ],
        'occlusion_2_unconstrained_no_imu': [
            '1VfpZexT629ptlfc8CRpLVaTVPcL4kC6X', '1LN2priMS3ynXt9d2H44jajCN6xLBi8Js'
        ],
        'cars_6_robot_no_imu': [
            '10Xbn1BlwXuPO2B37yvuDzUeMaKYHzn0B', '1IRHfAU-GuzoFdvZJS1xAc7kZaVBePCY4'
        ],
        'cars_6_unconstrained_no_imu': [
            '1lRXhGac90fl1bCNUmxavL7D1mSgfYqb3', '15N08qZFj_axl-GBzdcu2RrZrVvllS2E5'
        ],
        'cars_2_static_no_imu': [
            '1QGR1kF2SsEHtco2a-rnoCOqj9TdApl1R', '112RH4X1Q7Rxrrfg5QOlgOpykstyYTTlv'
        ],
        'cars_6_static_no_imu': [
            '1Ep59Nhp6xfDMOtcos4vAxYopG2ZqJssA', '1U3HsPk8MA3Dnf6S6iBlqmFyTdxvsZX-U'
        ],
    },
}

Oxford_Multimotion_calibration = {
    'calibration': {
        'files': [
            ('vicon_2019_06_14.yaml', '15akCf8l_Pg82HG6k1NT0gspoGTJss0zS'),
            ('manufacturer_2019_06_14.yaml', '1Zv2ru6Kl9-uKyzrLW7_lD93_VOpquP8R'),
            ('kalibr_2019_06_14.yaml', '19fIQm5UCM3kF3ZXn5nDdiFBZT65EqzEY'),
        ]
    },
}

def download_from_url(url: str, save_path: Path, overwrite: bool = False,
                      exclude_files: Optional[List[str]] = None,
                      exclude_dirs: Optional[List[str]] = None):
    subpath = Path(urlsplit(url).path)
    num_parents = len(subpath.parents)
    exclude = ['index.html*']
    if exclude_files is not None:
        exclude += exclude_files
    cmd = [
        'wget', '-r', '-np', '-nH', '-q', '--show-progress',
        '-R', f'"{",".join(exclude)}"',
        '--cut-dirs', str(num_parents), url,
        '-P', str(save_path)
    ]
    if exclude_dirs is not None:
        path = Path(urlsplit(url).path)
        cmd += ['-X', ' '.join(str(path / d) for d in exclude_dirs)]
    if not overwrite:
        cmd += ['-nc']
    print('Downloading ', url)
    subprocess.run(' '.join(cmd), check=True, shell=True)


def download_from_google_drive(id: str, save_path: Path,
                               chunk_size: int = 32768):
    url = 'https://docs.google.com/uc?export=download'
    session = requests.Session()
    response = session.get(url, params={'id': id}, stream=True)
    params = {'id': id, 'confirm': 1}
    response = session.get(url, params=params, stream=True)

    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def extract_tar(tarpath: Path, extract_path: Optional[Path] = None,
                remove: bool = True):
    if extract_path is None:
        extract_path = tarpath.parent
    print('Extracting ', tarpath)
    with tarfile.open(tarpath, 'r') as f:
        f.extractall(extract_path)
    if remove:
        tarpath.unlink()


def download_TUM_RGBD(datapath, sequences_to_download: Optional[list] = None):
    if sequences_to_download:
        sequences = sequences_to_download
    else:
        sequences = TUM_RGBD_Sequences
        configs = [k for k in sequences.keys()]
        sequences = [f'{c}/{seq}' for c in configs for seq in sequences[c]]
    
    url = URLs['TUM_RGBD']
    out_path = Path(datapath) / 'TUM_RGBD_Dataset'

    # Create a folder to hold the dataset if it doesn't exist
    out_path.mkdir(exist_ok=True, parents=True)
    print('Downloading the TUM RGB-D Dataset...')
    for sequence in sequences:
        # Check if sequence is already downloaded
        if os.path.exists(os.path.join(out_path, sequence[10:])):
            print(f"{sequence} already exists, skip...")
            continue
        download_from_url(url + f'{sequence}.tgz', out_path)
        extract_tar(out_path / f'{sequence[10:]}.tgz', out_path)


def download_Bonn_RGBD(datapath, sequences_to_download: Optional[list] = None):
    if sequences_to_download:
        sequences = sequences_to_download
    else:
        sequences = Bonn_RGBD_Sequences
        sequences = [f'{seq}' for seq in sequences["seq"]]
    
    url = URLs["Bonn_RGBD"]
    out_path = Path(datapath) / "Bonn_RGBD_Dataset"

    out_path.mkdir(exist_ok=True, parents=True)
    print('Downloading the Bonn RGB-D Dataset...')
    for sequence in sequence:
        # Check if sequence is already downloaded
        if os.path.exists(os.path.join(out_path, sequence)):
            print(f"{sequence} already exists, skip...")
            continue
        download_from_url(url + f'{sequence}.zip', out_path)
        extract_tar(out_path / f'{sequence}.zip', out_path)


def download_Oxford_Multimotion(datapath, sequences_to_download: Optional[list] = None):
    def find_sequences(sequences: list):
        keys = [seq.split('/') for seq in sequences]
        seq_dict = {f'{seq[0]}/{seq[1]}': Oxford_Multimotion_sequences[seq[0]][seq[1]] for seq in keys}
        return seq_dict

    if sequences_to_download:
        sequences = find_sequences(sequences_to_download)
    else:
        sequences = Oxford_Multimotion_sequences
        configs = [k for k in sequences.keys()]
        sequences = {f'{c}/{seq}': ids for c in configs for seq, ids in sequences[c].items()}
    
    out_path = Path(datapath) / "Oxford_Multimotion"
    out_path.mkdir(exist_ok=True, parents=True)
    print("Downloading the Oxford Multimotion Dataset...")
    for seq_name, ids in sequences.items():
        seq_outpath = os.path.join(out_path, seq_name)
        if os.path.exists(seq_outpath):
            print(f"{seq_name} already exists, skip...")
            continue
        Path(seq_outpath).mkdir(exist_ok=True, parents=True)
        if len(ids) == 3:
            outfiles = ['rgbd.tgz', 'imu.csv', 'vicon.csv']
        else:
            outfiles = ['rgbd.tgz', 'vicon.csv']
        for i in range(len(ids)):
            download_from_google_drive(
                ids[i], save_path=os.path.join(seq_outpath, outfiles[i])
            )
            if outfiles[i] == 'rgbd.tgz':
                extract_tar(
                    Path(seq_outpath) / outfiles[i], 
                    Path(seq_outpath) / 'rgbd'
                )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="TUM_RGBD", type=str, help="dataset to download")
    parser.add_argument("-s", "--sequences", default=[], nargs='+', help="sequences to download")
    parser.add_argument("-p", "--datapath", default='/home/jingkun/Dataset', type=str, help="directory to store the downloaded sequences")

    args = parser.parse_args()

    download_func = 'download_' + args.dataset
    download_func = locals()[download_func]

    if len(args.sequences) != 0:
        download_func(args.datapath, args.sequences)
    else:
        download_func(args.datapath)

    print("All sequences downloaded.")