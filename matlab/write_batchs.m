clc;clear;close all;

code_dir = '/share4/huoy1/Deep_5000_Brain/code/python_all_piece/';
epoch = 6;

for i = 1:3
    for j = 1:3
        for k=1:3
            piece = sprintf('%d_%d_%d',i,j,k);
            out_file = [code_dir filesep sprintf('t%s.sh',piece)];
            if ~exist(out_file)
                mid = fopen(out_file,'w');
                fprintf(mid,'#! /bin/bash \n');
                fprintf(mid,'#SBATCH --account=p_masi_gpu \n');
                fprintf(mid,'#SBATCH --partition=pascal \n');
                fprintf(mid,'#SBATCH --gres=gpu:1 \n');
                fprintf(mid,'#SBATCH --nodes=1 \n');
                fprintf(mid,'#SBATCH --ntasks=1 \n');
                fprintf(mid,'#SBATCH --mem=12G \n');
                fprintf(mid,'#SBATCH --time=70:00:00 \n');
                fprintf(mid,'#SBATCH --output=/scratch/huoy1/projects/DeepLearning/Deep_5000_Brain/accre_log/train_log_%s.txt \n',piece);
                
                fprintf(mid,'\n');
                fprintf(mid,'setpkgs -a anaconda3 \n');
                fprintf(mid,'source activate python27 \n');
                fprintf(mid,'setpkgs -a cuda8.0 \n');
                
                fprintf(mid,'\n');
                fprintf(mid,'python train.py --epoch=20 --piece=%s \n',piece);
                fclose(mid);
            end
        end
    end
end


















