"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


import os
import nibabel as nib
import util.cmr_dataloader as cmr
import util.cmr_transform as cmr_tran
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

# TR_CLASS_MAP_MMS_SRS= {'MYO': 2,'LV_Blood': 1, 'Scar': 3,'BG': 0,'NO_reflow': 4}
# TR_CLASS_MAP_MMS_DES= {'MYO': 2,'LV_Blood': 1, 'Scar': 3,'BG': 0,'NO_reflow': 4}
# TR_CLASS_MAP_MMS_SRS= {'MYO': 2,'LV_Blood': 1, 'Scar': 3,'NO_reflow': 4}
# TR_CLASS_MAP_MMS_DES= {'MYO': 2,'LV_Blood': 1, 'Scar': 3,'NO_reflow': 0}

# TR_CLASS_MAP_MMS_SRS= {'MYO': 2,'LV_Blood': 1, 'Scar': 3,'NO_reflow': 4}
# TR_CLASS_MAP_MMS_DES= {'MYO': 0,'LV_Blood': 1, 'Scar': 2,'NO_reflow': 3}
# sina feb 2021 for the heart with separated  labels
# TR_CLASS_MAP_MMS_SRS= {'BG': 0,'LV_Bloodpool': 8, 'LV_Myocardium': 9,'RV_Bloodpool': 10,'abdomen': 4,'Body_fat': 2,'vessel': 6, 'extra_heart': 1, 'Lung': 5,'Skeletal': 3}
# TR_CLASS_MAP_MMS_DES= {'BG': 0,'LV_Bloodpool': 1, 'LV_Myocardium': 2,'RV_Bloodpool': 3,'abdomen': 4,'Body_fat': 5,'vessel': 6, 'extra_heart': 7, 'Lung': 8 ,'Skeletal': 9}

TR_CLASS_MAP_MMS_SRS= {'BG': 0,'LV_Bloodpool': 1, 'LV_Myocardium': 2,'RV_Bloodpool': 3}
TR_CLASS_MAP_MMS_DES= {'BG': 0,'LV_Bloodpool': 1, 'LV_Myocardium': 2,'RV_Bloodpool': 3}

class Mms2BBDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
       
        # parser.set_defaults(label_nc=4)
        parser.set_defaults(output_nc=1)
        # parser.set_defaults(crop_size=128)
        parser.set_defaults(no_instance=True)
        parser.set_defaults(add_dist=False)
        

        parser.add_argument('--label_dir', type=str, required=False, default = '/data/sina/dataset/LGE_emidec/AI_image_data/3D_patients/with_normalization',
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, required=False, default ='/data/sina/dataset/LGE_emidec/AI_image_data/3D_patients/with_normalization' ,
                            help='path to the directory that contains photo images')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
                        
        return parser

    def get_paths(self, opt):
        """
        To prepare and get the list of files
        """
        if 'GE_BB' in self.opt.vendor:
            opt.main_dir = os.path.join(opt.image_dir, 'GE_BB')
        elif 'Philips_BB' in self.opt.vendor:
            opt.main_dir = os.path.join(opt.image_dir, 'Philips_BB')
        elif 'Siemens_BB' in self.opt.vendor:
            opt.main_dir = os.path.join(opt.image_dir, 'Siemens_BB')
        elif 'GE_LA' in self.opt.vendor:
            opt.main_dir = os.path.join(opt.image_dir, 'GE')
        elif 'Philips_LA' in self.opt.vendor:
            opt.main_dir = os.path.join(opt.image_dir, 'Philips')
        elif 'Siemens_LA' in self.opt.vendor:
            opt.main_dir = os.path.join(opt.image_dir, 'Siemens')
        else:
            raise ValueError('which vendor')


        assert os.path.exists(opt.main_dir), 'list of images doesnt exist'

        datalist = sorted(os.listdir(os.path.join(opt.main_dir)))
        LA_image_list = sorted(os.listdir(os.path.join(opt.main_dir, 'Image')))
        LA_mask_list = sorted(os.listdir(os.path.join(opt.main_dir, 'Label')))
        SA_image_list = sorted(os.listdir(os.path.join(opt.main_dir, 'Image')))
        SA_mask_list = sorted(os.listdir(os.path.join(opt.main_dir, 'Label')))
        

        # for subject in datalist:
        #     imagelist = sorted(os.listdir(os.path.join(opt.main_dir, subject)))
        #     for image in imagelist:
        #         if 'LA_ED.nii' in image or 'LA_ES.nii' in image:
        #             LA_image_list.append(image)
        #         if 'LA_ED_gt.nii' in image or 'LA_ES_gt.nii' in image:
        #             LA_mask_list.append(image)
        #         if 'SA_ED.nii' in image or 'SA_ES.nii' in image:
        #             SA_image_list.append(image)
        #         if 'SA_ED_gt.nii' in image or 'SA_ES_gt.nii' in image:
        #             SA_mask_list.append(image)

        assert len(LA_image_list) == len(LA_mask_list) 
        assert len(SA_image_list) == len(SA_mask_list)


        LA_filename_pairs = []
        SA_filename_pairs = [] 


        # for i in range(len(LA_image_list)):
        #     LA_filename_pairs += [(os.path.join(opt.main_dir,str(LA_image_list[i].split('_')[0]),LA_image_list[i]), os.path.join(opt.main_dir,str(LA_image_list[i].split('_')[0]),LA_mask_list[i]) )]
        #     if not opt.no_Short_axis:
        #         SA_filename_pairs += [(os.path.join(opt.main_dir,str(SA_image_list[i].split('_')[0]),SA_image_list[i]), os.path.join(opt.main_dir,str(SA_image_list[i].split('_')[0]),SA_mask_list[i]) )]
        # print('the size of the image list', len(SA_image_list))
        for i in range(len(SA_image_list)):
            SA_filename_pairs += [(os.path.join(opt.main_dir, 'Image',SA_image_list[i]), os.path.join(opt.main_dir,'Label',SA_mask_list[i]))]

        for i in range(len(LA_image_list)):
            LA_filename_pairs += [(os.path.join(opt.main_dir, 'Image',LA_image_list[i]), os.path.join(opt.main_dir,'Label',LA_mask_list[i]))]
                

        imglist = []
        msklist = []
        filename_pairs = []
        if 'LA' in self.opt.vendor:
            imglist = LA_image_list
            msklist = LA_mask_list
            filename_pairs = LA_filename_pairs
        if 'SA' in self.opt.vendor:
            imglist = SA_image_list
            msklist = SA_mask_list
            filename_pairs = SA_filename_pairs
        
    
        self.img_list = imglist
        self.msk_list = msklist
        self.filename_pairs = filename_pairs

        return self.filename_pairs, self.img_list, self.msk_list



    def initialize(self, opt):
        self.opt = opt
        if 'All' in self.opt.vendor:
            if 'LA' in self.opt.vendor:
                self.opt.vendor = 'GE_LA'
                self.filename_pairs_GE, _, _ = self.get_paths(self.opt)
                self.opt.vendor = 'Siemens_LA'
                self.filename_pairs_Siemens, _, _  = self.get_paths(self.opt)
                self.opt.vendor = 'Philips_LA'
                self.filename_pairs_Philips, _, _  = self.get_paths(self.opt)
                self.filename_pairs = self.filename_pairs_GE + self.filename_pairs_Siemens + self.filename_pairs_Philips
            elif 'SA' in self.opt.vendor:
                self.opt.vendor = 'SA_GE_BB'
                self.filename_pairs_GE, _, _ = self.get_paths(self.opt)
                self.opt.vendor = 'SA_Siemens_BB'
                self.filename_pairs_Siemens, _, _  = self.get_paths(self.opt)
                self.opt.vendor = 'SA_Philips_BB'
                self.filename_pairs_Philips, _, _  = self.get_paths(self.opt)
                self.filename_pairs = self.filename_pairs_GE + self.filename_pairs_Siemens + self.filename_pairs_Philips
            else:
                raise RuntimeError('choose a correct vendor data')
   
        else:
            self.filename_pairs, _, _  = self.get_paths(self.opt)


        print('the size of the image list', len(self.filename_pairs))

        if opt.phase == 'train':
            train_transforms = Compose([
                cmr_tran.Resample(self.opt.target_res,self.opt.target_res), #1.33
                cmr_tran.CenterCrop2D((self.opt.crop_size,self.opt.crop_size)),
                # cmr_tran.RandomRotation(degrees=90),
                # cmr_tran.RandomRotation(p=0.5),
                cmr_tran.ToTensor(),
                cmr_tran.PercentileBasedRescaling(out_min_max=(-1,1), percentiles=(1,99)),  #TODO: make sure the normalization is performed on the volume data not slice-by-slice
                # cmr_tran.RandomElasticTorchio(num_control_points  = (8, 8, 4), max_displacement  = (20, 20, 0), p=0.5),
                # cmr_tran.ClipScaleRange(min_intensity= 0, max_intensity=1),
                # cmr_tran.ClipTanh(),
                # cmr_tran.ClipScaleRange(),
                # cmr_tran.ClipNormalize(min_intensity= 0, max_intensity=4000),
                # cmr_tran.ClipZscoreMinMax(min_intensity= 0, max_intensity=4000),
                cmr_tran.RandomHorizontalFlip2D(p=0.50),
                cmr_tran.RandomVerticalFlip2D(p=0.50),
                cmr_tran.UpdateLabels(source=TR_CLASS_MAP_MMS_SRS, destination=TR_CLASS_MAP_MMS_DES)

            ])
        else:
            train_transforms = Compose([
                cmr_tran.Resample(self.opt.target_res,self.opt.target_res), #1.33
                cmr_tran.CenterCrop2D((self.opt.crop_size,self.opt.crop_size)),
                cmr_tran.RandomDilation_label_only(kernel_shape ='elliptical', kernel_size = 3, iteration_range = (1,2) , p=0.5),
                # cmr_tran.RandomRotation(degrees=90),
                # cmr_tran.RandomRotation(p=0.5),
                cmr_tran.ToTensor(),
                cmr_tran.PercentileBasedRescaling(out_min_max=(-1,1), percentiles=(1,99)),  #TODO: make sure the normalization is performed on the volume data not slice-by-slice
                cmr_tran.RandomElasticTorchio_label_only(num_control_points  = (8, 8, 4), max_displacement  = (14, 14, 1), p=1),
                
                # cmr_tran.RandomElasticTorchio(num_control_points  = (8, 8, 4), max_displacement  = (20, 20, 0), p=0.5),
                # cmr_tran.ClipScaleRange(min_intensity= 0, max_intensity=4000),
                # cmr_tran.ClipTanh(),
                # cmr_tran.ClipScaleRange(),
                # cmr_tran.RandomHorizontalFlip2D(p=0.5),
                # cmr_tran.RandomVerticalFlip2D(p=0.5),
                cmr_tran.UpdateLabels(source=TR_CLASS_MAP_MMS_SRS, destination=TR_CLASS_MAP_MMS_DES)

            ])
        
        self.cmr_dataset = cmr.MRI2DSegmentationDataset(self.filename_pairs, transform = train_transforms, slice_axis=2, canonical = False)
        
        
        size = len(self.cmr_dataset)
        self.dataset_size = size


    def __getitem__(self, index):
        # Label Image
        data_input = self.cmr_dataset[index]

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_tensor = data_input["gt"] # the label map equals the instance map for this dataset
        if not self.opt.add_dist:
            dist_tensor = 0
        input_dict = {'label': data_input['gt'],
                      'instance': instance_tensor,
                      'image': data_input['input'],
                      'path': data_input['filename'],
                      'gtname': data_input['gtname'],
                      'index': data_input['index'],
                      'segpair_slice': data_input['segpair_slice'],
                      'dist': dist_tensor
                      }

        return input_dict
    
    def __len__(self):
        return self.cmr_dataset.__len__()