import os,glob
import nilearn.input_data
import numpy


def get_masked_data(hyp,mask_img,output_dir,imgtype='unthresh',dataset='rectified'):
    if imgtype == 'unthresh':
        hmaps = glob.glob(os.path.join(output_dir,'%s/*/hypo%d_unthresh.nii.gz'%(dataset,hyp)))
    elif imgtype == 'thresh':
        hmaps = glob.glob(os.path.join(output_dir,'%s/*/hypo%d_thresh.nii.gz'%(dataset,hyp)))
    else:
        raise Exception('bad imgtype argument')
       
    #combined_data = nilearn.image.concat_imgs(hmaps)
    masker = nilearn.input_data.NiftiMasker(mask_img=mask_img)
    maskdata=masker.fit_transform(hmaps) #combined_data)   
    maskdata = numpy.nan_to_num(maskdata)
    if imgtype=='thresh':
        maskdata = (maskdata>1e-4).astype('float')
    labels = [os.path.basename(os.path.dirname(i)).split('_')[1] for i in hmaps]
    return(maskdata,labels)
