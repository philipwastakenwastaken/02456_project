from skimage.transform import rescale

# crop and rescale image
def transform(img):
    img = rescale(img, 0.25, anti_aliasing=False)*255
    img = img[5:36,2:-2] #asterix
    return img