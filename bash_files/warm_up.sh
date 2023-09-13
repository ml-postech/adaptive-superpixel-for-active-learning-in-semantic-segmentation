project_dir=/home/khy/github/Adaptive-Superpixels
export PYTHONPATH=${project_dir}/deeplab:${project_dir}/slim:${project_dir}/deeplab/datasets:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
dataset_name=cityscapes
model_name_seg=xception_65
region_num_per_image=8292
region_size=32
train_itr=60000
num_batch=1
base_learning_rate=0.007
train_split=train
job_name=8192_50k_v0_1
seed=1
crop_size=2049
k_array=(50000)
region_idx_dir=./region_index/$job_name
mkdir -p $region_idx_dir
anno_cost_dir=None
valid_idx_dir=./superpixels/$dataset_name/seeds_8192/train/label # change path for oracle
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
echo start batch now is $start_batch
for ((batch_id=start_batch;batch_id<num_batch;batch_id++));
 do
    if [ "$batch_id"  == 0 ]; then
       k=${k_array[$batch_id]}
    elif [ "$batch_id"  -lt "$num_batch" ]; then
       k=$(( k_array[batch_id] - k_array[batch_id-1] ))
    fi
    python ./scripts/region_selection_using_random.py \
                        --region_idx_dir=$region_idx_dir \
                        --list_folder=$list_folder \
                        --region_num_per_image=$region_num_per_image \
                        --batch_id=$batch_id \
                        --k=$k \
                        --train_split=$train_split \
                        --valid_idx_dir=$valid_idx_dir \
                        --anno_cost_dir=$anno_cost_dir \
                        --cost_type=rc \
                        --seed=$seed \
                        2>&1 | tee ./logs/${job_name}_${batch_id}_random.log               
    echo Generating batch training data...
    semantic_segmentation_folder_region=./deeplab/datasets/cityscapes/gtFineRegion/$job_name/batch_$batch_id
    mkdir -p $semantic_segmentation_folder_region
    tfrecord_dir=./deeplab/datasets/$dataset_name/tfrecord/$job_name/batch_$batch_id
    mkdir -p $tfrecord_dir
    python ./deeplab/build_data_active_sp.py \
                        --dataset_name=cityscapes \
                        --list_folder=$list_folder \
                        --tfrecord_dir=$tfrecord_dir \
                        --image_folder=$image_folder \
                        --semantic_segmentation_folder=$semantic_segmentation_folder \
                        --semantic_segmentation_folder_region=$semantic_segmentation_folder_region \
                        --region_idx_dir=$region_idx_dir \
                        --valid_idx_dir=$valid_idx_dir \
                        --batch_id=$batch_id \
                        --region_type=sp \
                        --train_split=$train_split \
                        --is_uniq=True  \
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
