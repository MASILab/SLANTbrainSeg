function copy_usable()
clc;clear;close all;

csv_path = '/home/local/VANDERBILT/xiongy2/Desktop/yunxi_imagevu_qa.csv';
raw_data_dir = '/share4/xiongy2/rawdata/SecondHalf';
input_dir = '/share5/xiongy2/INPUTS1';

table = readtable(csv_path, 'ReadRowNames', true);


for i = 549:945           
    subtable = table(i:i, {'ImageFile1', 'ImageFile1QA','ImageFile2','ImageFile2QA',...
        'ImageFile3','ImageFile3QA','ImageFile4','ImageFile4QA','ImageFile5',...
        'ImageFile5QA','ImageFile6','ImageFile6QA',	'ImageFile7','ImageFile7QA',...
        'ImageFile8','ImageFile8QA','ImageFile9','ImageFile9QA','ImageFile10','ImageFile10QA',...
        'ImageFile11','ImageFile11QA','ImageFile12','ImageFile12QA','ImageFile13',...
        'ImageFile13QA','ImageFile14','ImageFile14QA','ImageFile15','ImageFile15QA',...
        'ImageFile16','ImageFile16QA','ImageFile17', 'ImageFile17QA','ImageFile18','ImageFile18QA'});
    
    for i = 2 : 2 : 36
        name = char(subtable.(i-1));
        if (strcmp(subtable.(i),''))
            break
        end    
        subName = name(1:13);
        orig_file = [raw_data_dir filesep subName filesep name];
        target_file = [input_dir filesep name];
        if (strcmp(subtable.(i),'usable') | strcmp(subtable.(i), 'usable '))
            if ~isdir(target_file)
            system(sprintf('cp %s %s' ,orig_file,target_file));
            end
        end
    end
    
end    