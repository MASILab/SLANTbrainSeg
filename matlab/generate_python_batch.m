function generate_python_batch(normed_file,working_dir,model_dir,python_cmd,extra_dir)

if ~isdir(working_dir);mkdir(working_dir);end;
running_batch = [working_dir filesep 'test_all_pieces.sh'];
if ~exist(running_batch)
    fp = fopen(running_batch,'w');
    for x = 1:3
        for y = 1:3
            for z = 1:3
                piece = sprintf('%d_%d_%d',x,y,z);
                test_img_dir = [working_dir filesep 'deep_learning'];
                if ~isdir(test_img_dir);mkdir(test_img_dir);end;
                test_img_file = [test_img_dir filesep 'target_processed.nii.gz'];
                if ~exist(test_img_file);
                    system(sprintf('cp %s %s',normed_file,test_img_file));
                end
                
                out_dir = [working_dir filesep 'all_piece'];
                if ~isdir(out_dir);mkdir(out_dir);end;
                
                python_cmd_full = sprintf('%s %s/python/test.py',python_cmd,extra_dir);
                cmd = sprintf('%s --piece=%s --model_dir=%s --test_img_dir=%s --out_dir=%s\n',python_cmd_full,piece,model_dir,test_img_dir,out_dir);
                fprintf(fp,cmd);
                
            end
        end
    end
    fclose(fp);
    
end



end