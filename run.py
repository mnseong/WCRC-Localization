import argparse, logging
from pipeline import LocalizationPipeline
from pathlib import Path
from utils.evaluate import evaluate

logger = logging.getLogger(__name__)
SCENES = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']


def run_pipeline(images, gt_dir, retrieval, outputs, results, num_covis):
    # Instantiate the SceneReconstructor class with the given parameters
    run_scene = LocalizationPipeline(
        images=images,
        gt_dir=gt_dir,
        retrieval=retrieval,
        outputs=outputs,
        results=results,
        num_covis=num_covis,
    )

    # Set up the structure
    run_scene.setup_structure()

    # Extract local and global features
    run_scene.extract_features()

    # Perform Structure from Motion processes
    run_scene.perform_sfm()

    # Perform localization
    run_scene.localize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes', default=SCENES, choices=SCENES, nargs='+')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dataset', type=Path, default='datasets/7scenes',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='outputs/7scenes',
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--num_covis', type=int, default=30,
                        help='Number of image pairs for SfM, default: %(default)s')
    args = parser.parse_args()

    gt_dirs = args.dataset / '7scenes_sfm_triangulated/{scene}/triangulated'
    retrieval_dirs = args.dataset / '7scenes_densevlad_retrieval_top_10'

    all_results = {}
    for scene in args.scenes:
        results = args.outputs / scene / 'results.txt'
        if args.overwrite or not results.exists():
            run_pipeline(
                args.dataset / scene,
                Path(str(gt_dirs).format(scene=scene)),
                retrieval_dirs / f'{scene}_top10.txt',
                args.outputs / scene,
                results,
                args.num_covis
            )
        all_results[scene] = results

    for scene in args.scenes:
        logger.info(f'Evaluate scene "{scene}".')
        gt_dir = Path(str(gt_dirs).format(scene=scene))
        evaluate(gt_dir, all_results[scene], gt_dir / 'list_test.txt')
