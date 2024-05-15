# visudal studio


conda create -n calorie python=3.6

pip install -r requirements.txt --use-pep517 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install h5py==2.10 -i https://pypi.tuna.tsinghua.edu.cn/simple/

cd Food_volume_estimation


python volume_estimator.py --input_images ../Food_Detection/Myfood/images/test_set/kimbap/Img_069_0755.jpg --depth_model_architecture depth_architecture.json --depth_model_weights depth_weights.h5 --segmentation_weights segmentation_weights.h5











