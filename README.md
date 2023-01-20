# Domain adaptation for supervised integration of scRNA-seq data

The data used in the code can be found [here](https://drive.google.com/drive/folders/1vxet6GnzgonI0g7CxlrAAYdHzSNq-G8b). Please download it to a proper place and provide this path when you use a command to execute the main.py.

The data in our tutorial includes two files:myData_pancreatic_5batches.txt and mySample_pancreatic_5batches.txt. myData_pancreatic_5batches.txt is the concatenation of the gene expression matrix 5 batches to be integrated and mySample_pancreatic_5batches.txt is the corresponding metadata of the 5 batches with celltype and batch label inside it. 

All the package requirements are listed in requirement.txt

The demo script can be executed directly using a command in cmd
```
main.py --data_path=your_path_of_myData_pancreatic_5batches.txt --metadata_path=your_path_of_mySample_pancreatic_5batches.txt --epoch=15 --batch_size=512  --model_save_path=./your_model.h5 --siamese_g_save_path=your_siamese_model_g.h5
```
data_path and metadata_path are the paths for the input files we provided, the epoch and batch_size are the parameters during the training process, and the model_save_path is the path to save the whole SIDA model, the siamese_g_save_path is the path to save the shared siamese network g which is used to create a common embedding space in the SIDA structure. We provide the option to save this siamese network g just in case if someone need more following analysis based on it.

If you don't want to execute the demo with a command, train.py can be directly run to train a model, and the trained model will be saved under the project folder.

Pancreas_5dataset_all_cls.ipynb is a full instruction notebook, with training, visualization of embedding space and the leave-one-out cell type mapping experiments.
