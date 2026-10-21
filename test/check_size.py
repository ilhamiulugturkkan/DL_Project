from PIL import Image

im = Image.open("cars_train/100715345ee54d7ae38b52b4ee9d36a3/rgb/000000.png")  # use your real image path
print(im.size, im.mode)