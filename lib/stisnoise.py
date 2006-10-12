#!/usr/bin/env python

import pyfits
import numarray
import numarray.fft
import numarray.convolve
import math


__version__ = '5.3 (2003-Sep-09)'

def _median(arg):
    return numarray.sort(arg)[arg.getshape()[0]/2]

def medianfilter(time_series, width):
    tlen = time_series.shape[0]
    res = time_series.copy()
    res[:] = 0
    res[0] = time_series[0]
    beg, end = width/2, (width+1)/2
    for j in xrange(beg):
        res[j] = _median(time_series[:j+end])
    for j in xrange(beg, tlen-end):
        res[j] = _median(time_series[j-beg:j+end])
    for j in xrange(tlen-end, tlen):
        res[j] = _median(time_series[j-beg:])
    return res

def wipefilter(time_series, image_type, sst, freqmin, freqmax, scale):
    ntime = time_series.shape[0]
    # if ntime is an odd number the fft will take forever so make
    # it even with a small prime factor sum (not as quick as power
    # of 2 but still much quicker).
    # Note that padding data out to next power of two is too much
    # padding (number of elements is just a bit over 2^20 for STIS
    # data)
    if image_type == 'raw':
        ntimep = ntime+14
    else:
        ntimep = ntime+7
    t2    = numarray.zeros(ntimep, numarray.Float64)
    t2[:ntime] = time_series
    freq  = numarray.arange(ntimep)/(ntimep*sst*1.0e-6)
    freq[ntimep/2+1:ntimep] = freq[1:ntimep/2][::-1]
    tran  = numarray.fft.fft(t2)/len(t2)
    # apply filter
    ind   = numarray.nonzero((freq > freqmin)*(freq < freqmax))
    tran[ind] = tran[ind]*scale
    # inverse transform
    time_series = numarray.fft.inverse_fft(tran).real[:ntime+2]
    time_series *= time_series.shape[0]
    return time_series


def gauss(x, x0, dx, ymax):
    if dx > 0.:
        arg = numarray.clip(numarray.abs((x-x0)/dx), 0., 9.)
        y = numarray.exp(-arg*arg/2.)*(arg < 9.)
    else:
        y = (0.*x)*(x != x0)+(x == x0)
    return y*ymax

def windowfilter(time_series, image_type, sst, freqpeak, width, taper):
    import numarray.convolve
    ntime = time_series.shape[0]
    # if ntime is an odd number the fft will take forever so make
    # it even with a small prime factor sum (not as quick as power
    # of 2 but still much quicker).
    # Note that padding data out to next power of two is too much
    # padding (number of elements is just a bit over 2^20 for STIS
    # data)
    if image_type == 'raw':
        ntimep = ntime+14
    else:
        ntimep = ntime+7
    t2    = numarray.zeros(ntimep, numarray.Float64)
    t2[:ntime] = time_series
    freq  = numarray.arange(ntimep)/(ntimep*sst*1.0e-6)
    freq[ntimep/2+1:ntimep] = freq[1:ntimep/2][::-1]
    tran  = numarray.fft.fft(t2)/len(t2)
    # apply filter
    filter = numarray.ones(ntimep, numarray.Float64)
    ind   = numarray.nonzero((freq > (freqpeak-width/2.0)) * \
                             (freq < (freqpeak+width/2.0)))
    filter[ind] = 0.0
    freqstep = 1.0/(ntimep*sst*1.0e-6)
    width = taper/freqstep       # specify window width in freq steps
    sigma = width/2.354820044    # convert fwhm to sigma
    kernw = int(5*sigma)         # make kernel have width of 5 sigma
    if kernw%2 == 0:
        kernw = kernw+1          # make kernel odd
    kernx = numarray.arange(kernw)
    kerny = gauss(kernx, kernw/2, sigma, 1.0)  # gaussian kernel
    kerny = kerny/numarray.sum(kerny)
    filterc = numarray.convolve.correlate(filter, kerny, numarray.convolve.SAME)
    tran  = tran*filterc
    # inverse transform
    time_series = numarray.fft.inverse_fft(tran).real[:ntime+2]
    time_series *= time_series.shape[0]
    return time_series


def stisnoise(infile, exten=1, outfile=None, dc=1, verbose=1,
              boxcar=0, wipe=None, window=None):

    """
Description:

    Computes an FFT on STIS CCD frames to evaluate fixed pattern
    noise.  Fixed pattern noise is most obvious in a FFT of bias
    frames.  Optional filtering to correct the fixed pattern noise is
    provided through keywords boxcar, wipe, and window.  Filtered data
    can be saved as an output file.

Input:

    infile = STIS FITS file

Optional input (keywords):

    exten = fits extension to be read

    dc    = the power in the first freq bin is set to zero for better
    plotting of the power spectrum.

    verbose = set to 0 if you do not want brief information about each
    image.  Default is 1.

    Filtering options:

    boxcar = width of boxcar smoothing to be applied.

    wipe   = a 3-element array, specifying how to modify the data in
    frequency space. If set, the image is converted to a 1-D time
    series, fourier transformed to frequency space, modified, inverse
    transformed back to time space, and converted back to a 2-D image.
    The first and second elements specify the range in frequencies to
    be scaled (in hz), and the third element specifies the scaling
    factor (should be 0-1).

    window = a 3 element array, specifying how to modify the data in
    frequency space.  The first element is the center of the window
    (in hz). The second element is the width of the window (in hz).
    The third element controls the tapering of the window - it is the
    scale (in hz) of the tapering width.  Specifically, a square
    bandstop is convolved with a gaussian having the FWHM given by the
    third parameter.

Output:

    A tuple containing the arrays

    freq  = frequency in power spectrum (hz)
    magn  = magnitude in power spectrum

Optional output (keyword):

    outfile = filtered image

Author:

    Thomas M. Brown (STScI)

Python version:

    Paul Barrett (STScI)

"""

    # history:
    # 11/5/2001  TMB - version 1.  Basic idea comes from ACS analysis software
    #                  used for analyzing read noise 
    #                  (dino.pro; W.J. McCann & G. Hartig)
    # 11/6/2001  TMB - version 2 added other amps, error checking
    # 11/6/2001  TMB - version 2.1 added check on sci ext
    # 11/6/2001  TMB - version 3 added various filter options
    # 11/6/2001  TMB - version 3.1 added ability to read from STIS IDT DB
    # 11/7/2001  TMB - version 4 added scale filter and output images
    # 11/9/2001  TMB - version 4.1 added new parameter to scale routine,
    #                  changed
    #                  output to a file with header preservation.
    # 11/20/2001 TMB - version 5.0 added window routine, which does the
    #                  filtering of "scale" with a more gradual scaling
    # 11/26/2001 TMB - version 5.1 cleaned up the code comments.
    # 02/25/2002 JAV - version 5.2 added verbose option & output header
    #                  float type spec.
    # 05/21/2002 PEB - version 5.3 padded extra pixel with median of row.

    # Check filter options
    if ((boxcar > 0) + (wipe != None) + (window != None)) > 1:
        raise ValueError, 'conflicting filter options'

    # Define physical characteristics of STIS CCD
    pst = 640.0         # parallel shift time (us)
    sst = 22.0          # serial shift period (us)
    nc0 = 1062          # number of columns in raw data
    nr0 = 1044          # number of rows in raw data
    fltxy = 1024        # number of columns and rows in calibrated data
    nos = 19            # number of physical overscan columns
    pps = pst/sst       # number of serial shift intervals in parallel interval

    # Retrieve exposure information from header
    fin = pyfits.open(infile)
    extname = fin[exten].header['EXTNAME']
    inimage = fin[exten].data
    himage  = fin[0].data
    
    amp  = fin[0].header['CCDAMP']
    if verbose == 1:
        print 'Target: %s, Amp: %s, Gain: %d' % \
              (fin[0].header['TARGNAME'], amp, fin[0].header['CCDGAIN'])

    # Check to ensure the SCI extension is being used
    if extname != 'SCI':
        raise RuntimeError, \
              'You should only run this on a SCI extension, not %s.'%extname

    nr, nc = inimage.getshape()
    if   (nr, nc) == (nr0, nc0):
        image_type = 'raw'
    elif (nr, nc) == (fltxy, fltxy):
        image_type = 'flt'
    else:
        raise RuntimeError, 'This program should be run on 1062x1044 ' \
              'or 1024x1024 data only.'

    # Pad data with fake "OVERSCAN" if data have been overscan trimmed
    if image_type == 'flt':
        temp = numarray.zeros((fltxy, nc0), numarray.Float32)
        for row in range(fltxy):
            temp[row,:] = _median(inimage[row,:])
        temp[:,nos:nc0-nos] = inimage
        nc = nc0
    else:
        temp = inimage
    
    # Translate frame so that it is in readout order
    if   amp == 'A':
        image = temp             # amp A data -> leave as is
    elif amp == 'B':
        image = temp[::-1,:]     # amp B data -> flip left<->right
    elif amp == 'C':
        image = temp[:,::-1]     # amp C data -> flip top<->bottom
    elif amp == 'D':
        image = temp[::-1,::-1]  # amp D data -> rotate by 180 degrees
    else:
        raise RuntimeError, 'No amplifier given in header.'

    # Convert 2-D array to 1-D time series
    nx = nc + pps
    time_series = numarray.zeros(int(nx*nr), numarray.Float64)
    ds = numarray.zeros(int(pps), numarray.Float64)
    for i in range(nr):
        k = int(i*nx)
        # (note that non-integer nx prevents phase wandering)
        time_series[k:k+nc] = image[i,:]
        # pad dead-time
        medval = _median(image[i,:])
        time_series[k+nc:int(k+nc+pps)] = ds + medval
        if int((i+1)*nx) != int(k+nc+pps):
            time_series[int((i+1)*nx)-1] = medval

    # Begin filtering options ***************

    #if median != None:
    #    time_series = medianfilter(time_series, median)
    if boxcar > 0:
        time_series = numarray.convolve.boxcar(time_series, (boxcar,))

    elif wipe != None:
        time_series = wipefilter(time_series, image_type, sst,
                                 wipe[0], wipe[1], wipe[2])

    elif window != None:
        time_series = windowfilter(time_series, image_type, sst,
                                   window[0], window[1], window[2])

    # End filtering options ***************

    # Recreate 2-D image from time series
    outimage = numarray.zeros((nr, nc), numarray.Float32)
    for i in range(nr):
        outimage[i,:] = time_series[int(i*nx):int(i*nx+nc)]
    if image_type == 'flt':
        outimage      = outimage[:,nos:(nc0-nos)]

    # Restore original image orientation
    if   amp == 'A':
        pass                            # amp A data -> leave as is
    elif amp == 'B':
        outimage = outimage[::-1,:]     # amp B data -> flip left<->right
    elif amp == 'C':
        outimage = outimage[:,::-1]     # amp C data -> flip top<->bottom
    elif amp == 'D':
        outimage = outimage[::-1,::-1]  # amp D data -> rotate by 180 degrees

    # Trim vector to power of 2 for FFT 
    # (this is the fastest fft calculation but it doesn't preserve all
    # data, as needed in scale routine above)
    p2 = int(math.log(nx*nr)/math.log(2))
    n_ts = 2**p2
    time_series = time_series[:n_ts]

    # Perform FFT and return first half
    fft_output = numarray.fft.fft(time_series)/len(time_series)
    magnitude = numarray.abs(fft_output)[:n_ts/2]
    freq = numarray.arange(n_ts/2)/(n_ts*sst*1.0e-6)
    if dc == 1:
        # set first bin in power spectrum to zero if dc == 1
        magnitude[0] = 0

    if outfile:
        # write primary header then append ext
        fout = pyfits.HDUList()
        fout.append(pyfits.PrimaryHDU(header=fin[0].header))
        fout.append(pyfits.ImageHDU(header=fin[1].header, data=outimage))
        fout.writeto(outfile)
    
    return (freq, magnitude)

