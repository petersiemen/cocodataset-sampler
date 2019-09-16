# [COCO](http://cocodataset.org/#home) - 2014 Dataset - Sampler

The Microsoft COCO Dataset is widely used in Computer Vision research - not surprisingly it is quite huge.  

In order to be able to experiment nice and easy with your Neural Network Architectures  
you might want to have a fair sample of it that fits onto your notebook.

This python scripts generates a sample for/from the 2014 COCO dataset. 

## How to run the sampler
1. install python 3.6 and it's development files 
    > Ubuntu:
    ```bash 
    sudo apt-get install python3.6 python3.6-dev
    ```
2. install pip and pipenv
    ```bash
   curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
   python3.6 get-pip.py --user
   pip3.6 install --user pipenv
    ```
3. clone this repo
    ```bash
    git clone git@github.com:petersiemen/cocodataset-sampler.git
    cd cocodataset-sampler
    ``` 
4. install the samplers' dependencies in a pipenv
    ```bash
    pipenv install
    pipenv shell 
    ```
5. how to run the sampler
   ```bash
   python cocodataset_sampler.py 
   ```
   ```bash   
    usage: cocodataset-sampler [-h] [--annotations-dir ANNOTATIONS_DIR]
                               [--image-dir IMAGE_DIR] [--out-dir OUT_DIR]
                               [--n-per-category N_PER_CATEGORY]
    
    optional arguments:
      -h, --help            show this help message and exit
      --annotations-dir ANNOTATIONS_DIR
                            where to find the annotations' json files
      --image-dir IMAGE_DIR
                            where to find the image files
      --out-dir OUT_DIR     where to write to
      --n-per-category N_PER_CATEGORY
                            how many images to keep per category

    ```

   - it is assumed that the `IMAGE_DIR` (in the example below .../datasets/coco/images) has a layout like
    ```bash
   .../datasets/coco/images/train2014/...
   .../datasets/coco/images/val2014/...
   .../datasets/coco/images/test2014/...
    ```  
    - it is assumed that the `ANNOTATIONS_DIR` (in the example below .../datasets/coco/annotations) has a layout like
     ```bash
        .../datasets/coco/annotations/captions_train2014.json  
        .../datasets/coco/annotations/image_info_test2014.json  
        .../datasets/coco/annotations/instances_val2014.json           
        .../datasets/coco/annotations/person_keypoints_val2014.json
        .../datasets/coco/annotations/captions_val2014.json    
        .../datasets/coco/annotations/instances_train2014.json  
        .../datasets/coco/annotations/person_keypoints_train2014.json
      ```
6. optional: use the jupyter notebook to look at the sample

### Some Notes about COCO
#### Scene understanding
- recognize what objects are present
- localizing the objects in 2D and in 3D
- determining the objects' and scene's attributes
- characterizing the relationship between objects
- provide a semantic description of the scene

#### Objectives of COCO
- detecting non-iconic (or non-canonical perspectives) views of objects
- contectual reasoning between objects
- precise 2D localisation of objects

##### preliminary hypothesis:
- in order to push research in contextual reasoning, images depicting scenes rather than objects in isolation are necessary
- detailed spatial understanding of object layout will be a core component of scene analysis

#### COCO
- 91 common object catagories with 82 of them having more than 5,000 labeled instances
- 2,500,000 labeled instances in 328,000 images

