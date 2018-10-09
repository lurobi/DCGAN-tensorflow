import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from scipy.signal import convolve2d
import random
import string

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
    img = Image.fromarray(dat_int)
    img.save(fname)

def make_text(shape, w):
    dat_txt = np.zeros(shape,dtype=np.float32)
    img = Image.fromarray(dat_txt,'F')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 40)
    x = random.random()*shape[0]/2
    y = random.random()*shape[1]/2
    letter = random.choice(string.ascii_letters)
    angle = random.random() * 360

    draw.text((x,y),letter,1.0, font=font)

    img = img.rotate(angle,Image.BICUBIC)
    
    #img.save('tmp_img.png')
    dat_txt = np.frombuffer(img.tobytes(),dtype=np.float32).reshape(img.size)
    kern2d = make_kern(w)
    dat_noise = make_chi2(shape, 3, (0,0))
    dat_txt = dat_txt * dat_noise
    dat_txt = convolve2d(dat_txt,kern2d,"same","symm")
    return dat_txt

def make_kern(w):
    kern = np.hamming(w[0]).reshape(w[1],1)
    kern2d = np.matmul(kern,np.transpose(kern))
    return kern2d

def make_chi2(shape, k, w):
    dat = np.zeros(shape, dtype=np.float32)
    for jk in range(k):
        datk = np.float32(np.random.randn(shape[0], shape[1]))
        dat += datk**2

    if any(w):
        dat = convolve2d(dat,make_kern(w),"same","symm")
    return dat
    

def main():
    for jimg in range(128):
        fname = "data1/img_{}.png".format(jimg)
        make_one(fname,False)
def make_one(fname, diag=True):
    img_sz = 96
    if not fname: diag = True

    dat = make_chi2( (img_sz, img_sz), 3, (5,5))

    txt = make_text( (img_sz, img_sz), (5,5) )
    txt = (txt / txt.max()) * (random.random()+0.5)*2

    if diag:
        print('dat: {}, min={}, max={}'.format(dat.shape,dat.min(),dat.max()))
        #print('txt_noise: {}, min={}, max={}'.format(txt_noise.shape,txt_noise.min(),txt_noise.max()))
        print('txt: {}, min={}, max={}'.format(txt.shape,txt.min(),txt.max()))

    dat = dat / dat.mean()
    dat += txt

    dat = 10*np.log10(dat)

    if diag:
        write_img(dat,"noise.png",[0, 10])
        write_img(txt,"txt.png")
    if fname:
        write_img(dat,fname,[-2.5, 10])

if __name__ == "__main__":
    main()