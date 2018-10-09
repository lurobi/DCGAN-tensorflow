import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from scipy.signal import convolve2d

def write_img(dat, fname, clim=None):
    if clim == None:
        clim = (dat.min(), dat.max())
    print('File {} has clims of {}:{}'.format(fname,clim[0],clim[1]))
    dat = dat.copy()
    dat = dat - clim[0]
    dynamic_range = clim[1]- clim[0]
    dat = 255 * dat / dynamic_range
    dat = np.maximum(dat,0)
    dat = np.minimum(dat,255)
    dat = 255-dat # flip for black indicating "loud" pixels.
    dat_int = dat.astype(np.uint8,casting='unsafe')
    print(dat)
    img = Image.fromarray(dat_int)
    img.save(fname)

def make_text(shape):
    dat_txt = np.zeros(shape,dtype=np.float32)
    img = Image.fromarray(dat_txt,'F')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 40)
    draw.text((10,10),'A',1.5, font=font)
    #img.save('tmp_img.png')
    dat_txt = np.frombuffer(img.tobytes(),dtype=np.float32).reshape(img.size)
    #real_dat += dat_txt
    return dat_txt

def main():
    img_sz = 128

    dat = np.random.rand(img_sz, img_sz)
    dat = dat**2

    ksmth = 3
    kern = np.hamming(ksmth).reshape(ksmth,1)
    kern2d = np.matmul(kern,np.transpose(kern))
    
    ksmthtxt = 5
    kern_txt = np.hamming(ksmthtxt).reshape(ksmthtxt,1)
    kern2d_txt = np.matmul(kern_txt,np.transpose(kern_txt))
    
    #kern2d = kern2d / np.sqrt(np.mean(kern2d**2))


    print('dat: {}'.format(dat.shape))
    print('kern: {}, min={}, max={}'.format(kern2d.shape,kern2d.min(),kern2d.max()))

    dat = dat / dat.mean()

    dat_txt = make_text(dat.shape)


    dat = convolve2d(dat,kern2d,"same","symm")
    dat_txt = convolve2d(dat_txt,kern2d_txt,"same","symm")

    dat = dat / dat.mean()
    dat += dat_txt
    dat = 10*np.log10(dat)

    write_img(dat,"noise.png",[-5, 10])
    write_img(kern2d,"kernel.png")

main()