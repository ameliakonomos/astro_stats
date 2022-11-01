#!/usr/bin/env python
# coding: utf-8

# # Lab 3
# 
# Lab 3 centers on the photometric analysis of a short-period Cepheid variable star using a sequence of exposures in different filters, as well as associated calibration data.
# 
# As before, this notebook will first illustrate some Python features.

# In[1]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import pylab
import sys; sys.path.append('/home/a180i/lib/python/')
import a180
print('done')


# ## Python Features

# ### Tuple assignment
# 
# We normally assign values to a variable one at a time, like
# ~~~
# a = 10
# ~~~
# However, we can do more than one assignment at a time using tuples.  We could assign a tuple to the variable `a` like this:
# ~~~
# a = (5, 3)
# ~~~
# The variable `a` is now a tuple.  But what if we didn't want a tuple variable, what if we wanted two variables, `c` and `d` to have these two values?  We could do
# ~~~
# c = 5
# d = 3
# ~~~
# to assign them.  But using tuples, we can do it faster:

# In[2]:


c, d = (5, 3)  # tuple-based assignment
print(c,d)
c, d = 5, 3  # even less typing, since the tuple is implied
print(c,d)


# ### Defining functions
# 
# Sometimes we want to make our own functions.  Usually this occurs when we want to repeat a set of operations on a different set of variables.  (Note that when we want to repeat an operation on different values of variables, often we use loops.)
# 
# It is straightforward to create our own functions in Python.

# In[3]:


def myfunction(input1, input2):
    "my own function has a string to tell us what it does."
    output = input1 + input2
    return output
print(myfunction(2, 2))
print(myfunction(25490.2325, 09538294035.11))

def myotherfunction(input1, input2):
    "this one is more complicated.  we have two outputs"
    output1 = input1+input2
    output2 = input1*input2
    return output1, output2
out1, out2 = myotherfunction(2,3)
print(out1, out2)


# ### Functions with keyword arguments
# 
# Sometimes we can have optional parameters, where we can change the behavior of a function with that parameter, or have a default value for an input.  These are done with keywords arguments in Python.

# In[4]:


def anotherfunction(input1, optional_input=10):  # notice the syntax of the keyword argument.  We use an equals sign, and give the default value for when the function is called and the keyword is not present
    return input1+optional_input
print(anotherfunction(1))  # this function call will assume the default value for the keyword argument
print(anotherfunction(1, optional_input=2))  # notice how we call a function and specify the keyword


# ### Overlaying a circle on an image

# In[5]:


im = np.arange(32*32).reshape(32,32)  # some sample data to display as an image

# set up the figure
fig = pylab.figure()
ax = fig.add_subplot(111)

# show the image as grayscale
ax.imshow(im, cmap=mpl.cm.gray)

# overlay a circle
ax.add_patch(mpl.patches.Circle((16,10), # center of the circle (x,y)
                                radius=3., # radius of the circle
                                ec='r', # sets the circle edge color to blue
                                fill=False, # does not fill the circle with color
                               ))

# finalize the figure
pylab.draw()
pylab.show()


# ### Using +-, -=, *=, /= operators
# 
# Sometimes we want to modify a variable in place.  We might be tempted to do something like
# ~~~
# x = x + 2
# y = 3*y
# ~~~
# But there are special operators which can simplify this for us, the `+=`, `-=`, `*=`, and `/=` operators do in-place addition, subtraction, multiplication, and division.

# In[6]:


w, x, y, z = 1, 2, 3, 4  # multiple variable assignment
print(w, x, y, z)
w += 1
x -= 1
y *= 2
z /= 2
print(w, x, y, z)


# ### Plots with error bars

# In[7]:


# define some data
x = np.arange(5)
y = [1.2, .535, -1.4, 6.6, 12.]
y_err = [1., 2., 1., 0.5, 3.]

# set up the plot
fig = pylab.figure()
ax = fig.add_subplot(111)

# plot the data with errors
ax.errorbar(x, y, y_err)

# finalize the plot
pylab.show()


# ### Flipping the y axis of a plot
# 
# Sometimes we want to flip a $y$ axis, such as when plotting magnitudes (so bright things are higher in a graph).

# In[8]:


# set up the plot
fig = pylab.figure()
ax = fig.add_subplot(111)

# plot something
ax.errorbar(x, y, y_err)

# flip the y axes
ax.invert_yaxis()

# finalize the plot
pylab.show()


# ### Interpolation

# In[9]:


# get interpolation function
from scipy.interpolate import interp1d
yinterp_fn = interp1d(x, y, kind='cubic', bounds_error=False) # returns a function that, when evalued, gives interpolated values

# get values at which to interpolate
xint = np.linspace(0., 5., 100) # xmin, xmax, nx

# get interpolated values
yint = yinterp_fn(xint)

# set up the figure
fig = pylab.figure()
ax = fig.add_subplot(111)

# plot something
ax.errorbar(x, y, y_err, fmt='o')

# plot interpolated values
ax.plot(xint, yint)

# flip the y axes
ax.invert_yaxis()

# finalize the plot
pylab.show()


# ## Photometric Analysis of a Cepheid Variable Star
# 
# Let's get organized first, and specify our file locations.
# 
# We can't save our processed data in the raw data directory; we must specify a different location.  If it doesn't exist, we need to make it first.  You can do this from within Jupyter.  Suppose you want to make a directory called `foo`.  You would open a Terminal and type `mkdir foo`.

# In[10]:


raw_data_dir = '/data/home/a180f/rawlab3/' # directory where raw data are stored
proc_data_dir = '/data/home/a180f/processedlab3/' # place to store our processed data.  Make sure this exists!

# dark frames corresponding to the flats
raw_dark_files = ['dark001.FIT',
                  'dark002.FIT',
                  'dark003.FIT'
                 ]

# flat fields in V and R band
raw_flatV_files = ['domev001.FIT',
                   'domev002.FIT',
                   'domev003.FIT'
                  ]

raw_flatR_files = ['domer1.FIT',
                   'domer2.FIT',
                   'domer3.FIT'
                  ]


# sky images in V and R band
raw_skyV_files = ['skyV1.FIT'
                 ]

raw_skyR_files = ['skyR1.FIT'
                 ]

# science target observations in V and R band
raw_targV_files = ['v1.FIT',
                   'v2.FIT',
                   'v3.FIT',
                   'v4.FIT',
                   'v3.FIT',
                   'v4.FIT',
                   'v5.FIT',
                   'v6.FIT',
                   'v7.FIT',
                   'v8.FIT',
                   'v9.FIT',
                   'v10.FIT',
                   'v11.FIT',
                   'v12.FIT',
                   'v13.FIT',
                   'v14.FIT',
                   'v15.FIT',
                   'v16.FIT',
                   'v17.FIT',
                   'v18.FIT',
                   'v19.FIT',
                   'v20.FIT',
                   'v21.FIT',
                   'v22.FIT',
                   'v23.FIT',
                   'v24.FIT',
                   'v25.FIT',
                   'v26.FIT',
                   'v27.FIT'
                  ]

raw_targR_files = ['r1.FIT',
                   'r2.FIT',
                   'r3.FIT',
                   'r4.FIT',
                   'r3.FIT',
                   'r4.FIT',
                   'r5.FIT',
                   'r6.FIT',
                   'r7.FIT',
                   'r8.FIT',
                   'r9.FIT',
                   'r10.FIT',
                   'r11.FIT',
                   'r12.FIT',
                   'r13.FIT',
                   'r14.FIT',
                   'r15.FIT',
                   'r16.FIT',
                   'r17.FIT',
                   'r18.FIT',
                   'r19.FIT',
                   'r20.FIT',
                   'r21.FIT',
                   'r22.FIT',
                   'r23.FIT',
                   'r24.FIT',
                   'r25.FIT',
                   'r26.FIT',
                   'r27.FIT'
                  ]

# skies for the photometric standard stars 
raw_photskyV_files = ['skyv.FIT',
                      'skyv2.FIT',
                      'skyv3.FIT',
                      'skyv4.FIT',
                     ]

raw_photskyR_files = ['skyR1.FIT',
                      'skyr.FIT',
                      'skyr2.FIT',
                      'skyr3.FIT',
                      'skyr4.FIT'
                     ]

# observations of the photometric standard star (Landolt standard)
raw_photV_files = ['caliv001.FIT',
                   'caliv002.FIT',
                   'caliv003.FIT',
                   'caliv004.FIT',
                   'caliv005.FIT'
                  ]
raw_photR_files = ['cali1r.FIT',
                   'calir001.FIT',
                   'calir002.FIT',
                   'calir003.FIT',
                   'calir004.FIT'
                  ]

print('done')


# ### Calibrating the data
# 
# We'll need to calibrate our data.  For our target star images, we'll need to subtract out the background light using our sky exposures, as well as divide by the response function (the "flat field").
# 
# #### Creating the flat fields in each band
# 
# For each band ($V$ and $R$), we must first construct the flat field; this is done by removing the expected bias and dark current levels.
# 
# We're going to be loading and combining files fiarly often in our calibrations, so lets define a load and combine function.  We'll use median combination rather than averaging since it is more robust to errors that come from cosmic rays.
# 
# Let's first apply it to the dark exposure.

# In[15]:


from astropy.io import fits
import numpy as np

def load_and_combine(filenames, prefix=''):
    "Load and use median combination on a list of exposures.  Returns a numpy array."
    images = [] # define an empty list
    for fn in filenames:
        images.append(fits.getdata(prefix+fn)) # populate the list with image arrays from each file
    images = np.array(images) # turn the list into a 3d numpy array
    combined_im = np.median(images, axis=0) # use median combination along the first axis (image index)
    return combined_im

# process dark
dark_im = load_and_combine(raw_dark_files, prefix=raw_data_dir) # load and combine dark exposures into a dark frame
dark_fn = '/onedark.fits' # filename for our combined dark frame
onedark = fits.writeto(proc_data_dir+dark_fn, dark_im, overwrite=True) # store the combined dark frame in a FITS file
print('done')
print(np.shape(onedark))


# Now that we have a dark exposure, we can create the flat fields.
# 
# In a given band, we will load and combine the flat field exposures.  We will subtract out the dark frame, and then normalize by the median value so that the flat field frame represents a relative response level.  We can then save it to file.

# In[12]:


#our dark exposure is onedark.fits with path proc_data_dir 
im_fn = 'onedark.fits'
im = fits.getdata(proc_data_dir+im_fn) # this loads the image data from our FITS file into a variable

# create V-band flat field
v_flat = load_and_combine(raw_flatV_files, prefix=raw_data_dir)
v_fflat = (v_flat - im)
v_fflat_fn = 'vflat.fits'
v_flatv = fits.writeto(proc_data_dir+v_fflat_fn, v_flat, overwrite=True)

# create R-band flat field
r_flat = load_and_combine(raw_flatR_files, prefix=raw_data_dir)
r_fflat = (r_flat - im)
r_fflat_fn = 'rflat.fits'
r_flatr = fits.writeto(proc_data_dir+r_fflat_fn, r_flat, overwrite=True)

print('done')


# #### Processing sky exposures
# 
# We won't have to do anything fancy to the sky background exposures, just load them and combine them for each set.

# In[13]:


# process and store sky exposures for target, V band
# raw_photskyV_files
v_sky = load_and_combine(raw_photskyV_files, prefix=raw_data_dir)
v_sky_fn = 'v_sky.fits'
v_skyv = fits.writeto(proc_data_dir+v_sky_fn, v_sky, overwrite=True)

# process and store sky exposures for target, R band
# raw_photskyR_files 
r_sky = load_and_combine(raw_photskyR_files, prefix=raw_data_dir)
r_sky_fn = 'r_sky.fits'
r_skyr = fits.writeto(proc_data_dir+r_sky_fn, r_sky, overwrite=True)

# process and store sky exposures for photometric standard, V band
# raw_photV_files
v_sky_std = load_and_combine(raw_photV_files, prefix=raw_data_dir)
v_sky_stdfn = 'v_sky_std.fits'
v_sky_stdv = fits.writeto(proc_data_dir+v_sky_stdfn, v_sky_std, overwrite=True)
                   
# process and store sky exposures for photometric standard, R band
# raw_photR_files
r_sky_std = load_and_combine(raw_photR_files, prefix=raw_data_dir)
r_sky_stdfn = 'r_sky_std.fits'
r_sky_stdr = fits.writeto(proc_data_dir+r_sky_stdfn, r_sky_std, overwrite=True)

print('done')
print(v_skyv)


# #### Calibrating photometric exposures
# 
# We have several data sets we need to calibrate, namely our target and photometric standard star exposures in each filter band.
# 
# To do this, we'll load each exposure, subtract the sky background, and divide by the flat field response.  We'll store the output as a file.  Let's try to simplify the process by writing a function.

# In[14]:


def process_photometry(raw_filename, sky_im, flat_im, raw_dir='', out_dir='', prefix='proc-'):
    "Calibriate a photometric exposure"
    out_fn = prefix + raw_filename  # output filename
    im = fits.getdata(raw_dir+raw_filename)   # load the input data
    proc_im = (im - sky_im) / flat_im  # calibration steps
    fits.writeto(proc_data_dir+out_dir+out_fn, proc_im)  # store the output
    return out_fn  # return the processed image filename

# process photometric standard exposures, V band
proc_targ_V = []
for file in raw_photV_files:
    photometric_exp_v = process_photometry(file, v_skyv, v_flatv, raw_dir='/data/home/a180f/rawlab3/', out_dir='/data/home/a180f/processedlab3/', prefix='proc-v')
    proc_targ_V.append(photometric_exp_v)
                   
# process photometric standard exposures, R band
raw_photR_files 

# process target star exposures, V band
raw_targV_files 

# process target star exposures, R band
raw_targR_files 


# Now all our data have had the first level of calibration performed, and we're ready to start making photometric measurements.
# 
# ### Getting uncalibrated photometry
# 
# We need to add up the light from our measurements.  This will give us uncalibrated photometry, in units of DN/s.
# 
# To do this we'll define an "aperture," which is a virtual region in the image over which we'll add up the counts.  This will be a circular aperture centered on the star.
# 
# We also expect that our sky background subtraction is not perfect, so we'll define a "sky annulus," a ring-like region outside of our photometric aperture over which the residual sky level will be determiend and subtracted from our target aperture.
# 
# So we'll need to find:
# * star center in pixel coordinates
# * radius of the photometric aperture
# * inner and outer radii of the sky annulus
# 
# To simplify the analysis, we'll use the same aperture parameters on all exposures.  That way we only have to find the stellar centers in each exposure.
# 
# Let's start with an example exposure.
# 
# #### Example exposure photometry
# 
# Let's load and display an example exposure.  We'll overlay the photometric and sky annulus apertures on the image to check consistency, and we'll get the photometric measurement.

# In[ ]:


# first some definitions
proc_targ_im_fn = 'FIXME.fits'  # filename for processed example target exposure
t_exp = FIXME  # [s] exposure time
x, y = FIXME, FIXME  # [pix], [pix]  stellar x and y postiions
phot_rad = FIXME  # [pix] photometric aperture radius
sky_ann_inner_rad = FIXME  # [pix] sky annulus inner radius
sky_ann_outer_rad = FIXME  # [pix] sky annulus outer radius

# load the target image
im = FIXME

# create a figure
fig = pylab.figure()
ax = fig.add_subplt(111)

# display the image
ax.imshow(im)

# overlay a circle for the photometric aperture
ax.add_patch(mpl.patches.Circle((x,y), # center of the circle
                                radius=phot_rad, # radius of the circle
                                ec='b', # sets the circle edge color to blue
                                fill=False, # does not fill the circle with color
                               ))

# overlay a circle for the sky annulus inner radius

# overlay a circle for the sky annulus outer radius


# show the figure
pylab.draw()


# We should be sure the apertures are centered on the star.
# 
# The ideal size for the photometric aperture is roughly to have it as large as possible without being so big as being dominated by sky background noise.  So we should have it sized so that it encompasses the majority of the visible starlight.
# 
# Note that since we are using the same aperture sizes for all exposures, it won't be optimal in all cases.  We're aiming for "good enough" here.
# 
# The sky annulus inner radius should be large enough so that no signal from the star is in the sky annulus.  The outer radius should be large enough so that a good number of pixels are included (ideally more area in the sky annulus than the target aperture), but not so large that systematic errors from sky nonuniformity creep in.  There is no grea recipe for the sky annulus; just get something good enough.
# 
# Once we have our aperture set, let's do the photometry.

# In[ ]:


from a180 import ap_phot

gain = FIXME  # [e-/DN] gain of image sensor

phot, phot_err = ap_phot(im, x, y, 
                         phot_rad, 
                         sky_in=sky_ann_inner_rad, 
                         sky_out=sky_ann_outer_rad, 
                         gain=gain)  # get the aperture photometry
phot /= t_exp  # [DN] -> [DN/s]
phot_err /= t_exp  # [DN] -> [DN/s]

print(phot, phot_err)


# #### Positions for all star exposures
# 
# We need to get the star position for all of our exposures.  The simplest way is through visual inspection.

# In[ ]:


# define xy positions for target V-band exposures
targV_xys = [(FIXME, FIXME),
             (FIXME, FIXME),
            ]

# define xy positions for target R-band exposures


# define xy positions for standard V-band exposures


# define xy positions for standard R-band exposures



# #### Photometry for all exposures
# 
# Let's loop and get photometric measurements for all exposures.  Try writing a function to loop over an exposure sequence, and return numpy arrays of the photometry and photometric error, each in DN/s.

# In[ ]:


def FIXME():
    
    FIXME
    
    F /= t  # [DN] -> [DN/s]
    F_err /= t  # [DN] -> [DN/s]

    return F, F_err

F_V_targ, F_V_targ_err = FIXME()
F_R_targ, F_R_targ_err = FIXME()
F_V_std, F_V_std_err = FIXME()
F_R_std, F_R_std_err = FIXME()


# #### Plotting the raw photometry vs. time
# 
# Let's do a quick check and examine our photometry (and errors) vs. time in each band.

# In[ ]:


# target time offsets in V band
t_offs_V = [FIXME,
            FIXME,
           ]
# target time offsets in R band


# set up figure
fig = pylab.Figure()
ax1 = fig.add_subplot(121) # one row, two columns, first subplot
ax2 = fig.add_subplot(122) # one row, two columns, second subplot

# plot V band
ax1.errorbar(t_offs_V, F_V_targ, F_V_targ_err)
ax1.set_title('V')
ax1.set_xlabel('time offset [s]')
ax1.set_ylabel('photometry [DN/s]')

# plot R band



# finalize plot
pylab.draw()


# #### Calibrating the photometry
# 
# We'll use the standard star photometry and its known magnitudes in each filter to calibrate our starget star photometry and put it into magnitude units.
# 
# Let's first get an average DN/s level for our standard star in each band, along with an uncertainty.  

# In[ ]:


F0_V = FIXME  # [DN/s]
F0_V_err = FIXME  # [DN/s]
F0_R = FIXME  # [DN/s]
F0_R_err = FIXME  # [DN/s]


# Now we'll use this observed flux level and the known magnitude to get a zero point (and associated uncertainty).

# In[ ]:


m0_V = FIXME  # [mag]
m0_V_err = FIXME  # [mag]
m0_R = FIXME  # [mag]
m0_R_err = FIXME  # [mag]


# Finally we can use these to calibrate our target star photometry.

# In[ ]:


# calibrated target star V-band photometry [mag] with uncertainty


# calibrated target star R-band photometry [mag] with uncertainty



# #### Plotting the calibrated photometry
# 
# We'll want to plot our photometry (and errors) vs. time.

# In[ ]:





# #### Computing a color
# 
# We'll want to look at $V-R$ color vs. time as well.  One complication is that we don't have the same timestamps for our $V$ and $R$ exposures, so we can't just subtract them.
# 
# We can get a sense of the color change by interpolating the time sequences.
# 

# In[ ]:


# interpolate the time series data
from scipy.interpolate import interp1d
interp_mR_func = FIXME
interp_mR = interp_mR_func(t_offs_V) # interpolated magnitudes at V-band time locations

# compute the V-R color (magnitudes)


# estimate the uncertainties for the color




# #### Plotting the color
# Let's see the results.

# In[ ]:




