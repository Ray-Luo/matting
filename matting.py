import argparse


def parse_options(parser):
    parser.add_argument(
        "--mask_path",
        type=str,
        help="mask image path",
    )
    parser.add_argument(
        "--guidance_path",
        type=str,
        help="guidance image path",
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        help="output image path",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
        help="smoothing strength; larger means more smooothing, less detail",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=6,
        help="radius for searching near boundary, larger radius means searching for more potential fine strctures",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=1.,
        help="downscale images for fast processing",
    )
    parser.add_argument(
        "--is_mask_rgb",
        type=bool,
        default=False,
        help="if input mask is rgb",
    )
    parser.add_argument(
        "--is_guidance_rgb",
        type=bool,
        default=True,
        help="if guidance image is rgb",
    )
    return parser.parse_args()


def main():
    parser = argparse.ArgumentParser(description="Alpha matting from mask and guidance images")
    opt = parse_options(parser)

if __name__ == "__main__":
    main()
