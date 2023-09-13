project_dir=/home/khy/github/Adaptive-Superpixels
export PYTHONPATH=${project_dir}/deeplab:${project_dir}/slim:${project_dir}/deeplab/datasets:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
dataset_name=cityscapes
model_name_seg=xception_65
region_num_per_image=8292
num_superpixels=8192
region_size=32
train_itr=1000
num_batch=2
base_learning_rate=0.007
train_split=train
job_name=8192_50k_v1_1
knee=20
base_job_name=8192_50k_v0_1
random_ref_job=8192_50k_v0_1
alpha=0
crop_size=2049
k_array=(50000 100000)
region_idx_dir=./region_index/$job_name
mkdir -p $region_idx_dir
reg_idx_dir_ref=./region_index/$random_ref_job
anno_cost_dir=None
valid_idx_dir_0=/hdd/hdd4/khy/Revisiting/superpixels/$dataset_name/seeds_8192/train/label # check
valid_idx_dir_1=/hdd/hdd4/khy/Revisiting/superpixels/$dataset_name/256_50k_v1_1_256_v0_1_am_0.1 # check 
valid_idx_dir=/hdd/hdd4/khy/Revisiting/superpixels/$dataset_name/256_50k_v1_1_256_v0_1_am_0.1/batch_0/label # check !!
devkit_path=./deeplab/datasets/cityscapes/
list_folder=./deeplab/datasets/cityscapes/image_list
image_folder=./deeplab/datasets/cityscapes/leftImg8bit
semantic_segmentation_folder=./deeplab/datasets/cityscapes/gtFine
PATH_TO_INITIAL_CHECKPOINT=deeplab/models/$model_name_seg/model.ckpt
eval_data_dir=deeplab/datasets/cityscapes/tfrecord
mkdir -p ./accuracy_log
accuracy_log=./accuracy_log/${job_name}.txt
mkdir -p ./batch_log
batch_log=./batch_log/${job_name}.txt
mkdir -p ./logs
if test -f $batch_log; then
   typeset -i start_batch=$(cat $batch_log)
   echo start batch log is $start_batch
   start_batch=$(( start_batch + 1 ))
else
   start_batch=0
fi
if [ "$start_batch"  == 0 ]; then
    train_dir=./outputs/$job_name/batch_0 # check
    mkdir -p $train_dir
    train_dir_ref=./outputs/$random_ref_job/batch_0 # check
    if [ -d $train_dir_ref ]
    then
      cp  $train_dir_ref/frozen_inference_graph.pb $train_dir
    fi
    reg_idx_dir_ref=./region_index/$random_ref_job
    if [ -d $reg_idx_dir_ref ]
    then
      cp  $reg_idx_dir_ref/batch_0.pkl $region_idx_dir # check 
      cp  $reg_idx_dir_ref/batch_0_selected_idx.pkl $region_idx_dir # check
    fi
    start_batch=$(( start_batch + 1 )) # check
fi
echo start batch now is $start_batch
for ((batch_id=start_batch;batch_id<num_batch;batch_id++));
 do
    if [ "$batch_id"  == 0 ]; then
       k=${k_array[$batch_id]}
    elif [ "$batch_id"  -lt "$num_batch" ]; then
       k=$(( k_array[batch_id] - k_array[batch_id-1] ))
    fi
    region_uncert_dir=./region_uncertainty/$job_name/batch_$(( batch_id - 1 ))
    mkdir -p $region_uncert_dir
    class_to_region_idx_path=./class_to_region_idx/$job_name/batch_$(( batch_id - 1 ))/ctr_idx.pkl
    python scripts/extract_model_predictions_cb.py \
                        --dataset_name=$dataset_name \
                        --job_name=$job_name \
                        --batch_id=$(( batch_id - 1 )) \
                        --region_type=sp \
                        --region_num_per_image=$region_num_per_image \
                        --num_superpixels=$num_superpixels \
                        --region_size=32 \
                        --is_bal=True \
                        --sp_method=seeds \
                        --superpixel_label_dir=$valid_idx_dir \
                        2>&1 | tee ./logs/${job_name}_${batch_id}_model_pred.log
    python scripts/region_selection_using_cb.py \
                        --batch_id=$batch_id \
                        --list_folder=$list_folder \
                        --region_uncert_dir=$region_uncert_dir \
                        --region_idx_dir=$region_idx_dir \
                        --k=$k \
                        --region_num_per_image=$region_num_per_image \
                        --train_split=$train_split \
                        --region_size=$region_size \
                        --valid_idx_dir=$valid_idx_dir \
                        --anno_cost_dir=$anno_cost_dir \
                        --cost_type=rc \
                        --is_bal=True \
                        --class_to_region_idx_path=$class_to_region_idx_path \
                        2>&1 | tee ./logs/${job_name}_${batch_id}_al.log
    echo Generating batch training data...
    semantic_segmentation_folder_region=./deeplab/datasets/cityscapes/gtFineRegion/$job_name/batch_$batch_id
    semantic_segmentation_folder_region_prev=./deeplab/datasets/cityscapes/gtFineRegion/$base_job_name/batch_0
    mkdir -p $semantic_segmentation_folder_region
    tfrecord_dir=./deeplab/datasets/$dataset_name/tfrecord/$job_name/batch_$batch_id
    mkdir -p $tfrecord_dir
    python ./deeplab/build_data_active_sp_am.py \
                        --dataset_name=cityscapes \
                        --list_folder=$list_folder \
                        --tfrecord_dir=$tfrecord_dir \
                        --image_folder=$image_folder \
                        --semantic_segmentation_folder=$semantic_segmentation_folder \
                        --semantic_segmentation_folder_region=$semantic_segmentation_folder_region \
                        --semantic_segmentation_folder_region_prev=$semantic_segmentation_folder_region_prev \
                        --region_idx_dir=$region_idx_dir \
                        --valid_idx_dir_0=$valid_idx_dir_0 \
                        --valid_idx_dir_1=$valid_idx_dir_1 \
                        --batch_id=$batch_id \
                        --region_type=sp \
                        --train_split=$train_split \
                        --is_uniq=True \
                        --job_name=$job_name \
                        --knee=$knee \
                        2>&1 | tee ./logs/${job_name}_${batch_id}_build_data.log 
    train_dir=./outputs/$job_name/batch_$batch_id
    mkdir -p $train_dir
    echo Active training batch $batch_id ...
    python ./deeplab/train.py \
                        --logtostderr \
                        --training_number_of_steps=$train_itr \
                        --base_learning_rate=$base_learning_rate \
                        --num_clones=1 \
                        --train_split=$train_split \
                        --model_variant=$model_name_seg \
                        --train_crop_size=769 \
                        --train_crop_size=769 \
                        --atrous_rates=6 \
                        --atrous_rates=12 \
                        --atrous_rates=18 \
                        --output_stride=16 \
                        --decoder_output_stride=4 \
                        --train_batch_size=4 \
                        --dataset=$dataset_name \
                        --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \
                        --train_logdir=$train_dir \
                        --dataset_dir=$tfrecord_dir \
                        --fine_tune_batch_norm=True \
                        2>&1 | tee ./logs/${job_name}_${batch_id}_train.log
    python ./deeplab/export_model.py \
                        --logtostderr \
                        --checkpoint_path=$train_dir/model.ckpt-$train_itr \
                        --export_path=$train_dir/frozen_inference_graph.pb \
                        --model_variant=$model_name_seg \
                        --atrous_rates=6 \
                        --atrous_rates=12 \
                        --atrous_rates=18 \
                        --output_stride=16 \
                        --decoder_output_stride=4 \
                        --num_classes=19 \
                        --crop_size=512 \
                        --crop_size=1024 \
                        --inference_scales=1.0
    python ./deeplab/eval_once.py \
                        --checkpoint_path=$train_dir/model.ckpt-$train_itr \
                        --dataset=cityscapes \
                        --eval_logdir=$train_dir \
                        --dataset_dir=$eval_data_dir \
                        --model_variant=$model_name_seg \
                        --eval_crop_size=1025 \
                        --eval_crop_size=2049 \
                        --atrous_rates=6 \
                        --atrous_rates=12 \
                        --atrous_rates=18 \
                        --output_stride=16 \
                        --decoder_output_stride=4 \
                        --accuracy_log=$accuracy_log \
                        --batch_log=$batch_log \
                        --batch_id=$batch_id 
    if [ ! -f $accuracy_log ] 
    then
       echo training $batch_id is not successful 
       break 
    else
       rm -r $tfrecord_dir
    fi
done