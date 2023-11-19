import logging
import pathlib

import numpy as np

from core import extract, match, localize_sfm, triangulation, image_retrieval
from utils.read_write_model import read_cameras_binary, read_images_binary, read_model, write_model, read_images_text, read_cameras_text

logger = logging.getLogger(__name__)


class LocalizationPipeline:
    def __init__(self, images, gt_dir, retrieval, outputs, results, num_covis):
        # Initialization of the class variables
        self.sfm_matches = None
        self.global_features = None
        self.local_features = None
        self.test_list = None
        self.query_list = None
        self.sfm_pairs = None
        self.ref_sfm = None
        self.ref_sfm_sift = None
        self.images = images
        self.gt_dir = pathlib.Path(gt_dir)
        self.retrieval = retrieval
        self.outputs = pathlib.Path(outputs)
        self.results = results
        self.num_covis = num_covis

        # Configuration for local and global features
        self.local_feature_conf = extract.configs['superpoint']
        self.global_feature_conf = extract.configs['cosplace']

        # Configuration for matching
        self.matcher_conf = match.configs['superglue']
        self.matcher_conf['model']['sinkhorn_iterations'] = 5

        # Setting up necessary paths and reference structures
        self.setup_structure()

    def create_reference_sfm(self, full_model, ref_model, blacklist=None, ext='.bin'):
        # Create a new COLMAP model with only training images
        logger.info('Creating the reference model.')
        ref_model.mkdir(exist_ok=True)
        cameras, images, points3D = read_model(full_model, ext)

        if blacklist is not None:
            with open(blacklist, 'r') as f:
                blacklist = f.read().rstrip().split('\n')

        images_ref = dict()
        for id_, image in images.items():
            if blacklist and image.name in blacklist:
                continue
            images_ref[id_] = image

        points3D_ref = dict()
        for id_, point3D in points3D.items():
            ref_ids = [i for i in point3D.image_ids if i in images_ref]
            if len(ref_ids) == 0:
                continue
            points3D_ref[id_] = point3D._replace(image_ids=np.array(ref_ids))

        write_model(cameras, images_ref, points3D_ref, ref_model, '.bin')
        logger.info(f'Kept {len(images_ref)} images out of {len(images)}.')

    def create_query_list_with_intrinsics(model, out, list_file=None, ext='.bin', image_dir=None):
        if ext == '.bin':
            images = read_images_binary(model / 'images.bin')
            cameras = read_cameras_binary(model / 'cameras.bin')
        else:
            images = read_images_text(model / 'images.txt')
            cameras = read_cameras_text(model / 'cameras.txt')

        name2id = {image.name: i for i, image in images.items()}
        if list_file is None:
            names = list(name2id)
        else:
            with open(list_file, 'r') as f:
                names = f.read().rstrip().split('\n')
        data = []
        for name in names:
            image = images[name2id[name]]
            camera = cameras[image.camera_id]
            w, h, params = camera.width, camera.height, camera.params

            if image_dir is not None:
                # Check the original image size and rescale the camera intrinsics
                img = cv2.imread(str(image_dir / name))
                assert img is not None, image_dir / name
                h_orig, w_orig = img.shape[:2]
                assert camera.model == 'SIMPLE_RADIAL'
                sx = w_orig / w
                sy = h_orig / h
                assert sx == sy, (sx, sy)
                w, h = w_orig, h_orig
                params = params * np.array([sx, sx, sy, 1.])

            p = [name, camera.model, w, h] + params.tolist()
            data.append(' '.join(map(str, p)))
        with open(out, 'w') as f:
            f.write('\n'.join(data))

    def setup_structure(self):
        self.outputs.mkdir(exist_ok=True, parents=True)
        self.ref_sfm_sift = self.outputs / 'sfm_sift'
        self.ref_sfm = self.outputs / 'sfm_res'
        self.sfm_pairs = self.outputs / f'pairs-db-retrieval.txt'
        self.query_list = self.outputs / 'query_list.txt'
        self.test_list = self.gt_dir / 'list_test.txt'

        # Creation of reference structures and query lists
        self.create_reference_sfm(self.gt_dir, self.ref_sfm_sift, self.test_list)
        self.create_query_list_with_intrinsics(self.gt_dir, self.query_list, self.test_list)

    def extract_features(self):
        # Extract local and global features
        self.local_features = extract.main(
            self.local_feature_conf, self.images, self.outputs, as_half=True)
        self.global_features = extract.main(
            self.global_feature_conf, self.images, self.outputs, as_half=True)

    def perform_sfm(self):
        # Structure from Motion processes
        sfm_pairs = self.outputs / f'pairs-db-retrieval.txt'
        image_retrieval.main(descriptors=self.global_feature_conf, output=self.sfm_pairs, num_matched=20)

        self.sfm_matches = match.main(
            self.matcher_conf, sfm_pairs, self.local_feature_conf['output'], self.outputs)

        triangulation.main(
            self.ref_sfm, self.ref_sfm_sift,
            self.images,
            sfm_pairs,
            self.local_features,
            self.sfm_matches)

    def localize(self):
        # Localization process
        loc_matches = match.main(
            self.matcher_conf, self.retrieval, self.local_feature_conf['output'], self.outputs)

        localize_sfm.main(
            self.ref_sfm,
            self.query_list,
            self.retrieval,
            self.local_features,
            loc_matches,
            self.results,
            covisibility_clustering=True,
            prepend_camera_name=True)
