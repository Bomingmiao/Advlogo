from aesthetic import make_model
from PIL import Image
import clip
import argparse


def main(args):
    clip_model, preprocess = clip.load("ViT-L/14")
    aes_model = make_model()
    img = Image.open(args.img_path)
    img = preprocess(img)
    f = clip_model.encode_image(img.unsqueeze(0).cuda())
    aes_score = aes_model.get_aesthetic_score(f)
    print(aes_score)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path',type=str,default=None)
    args = parser.parse_args()
    main(args)