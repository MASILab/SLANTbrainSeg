function sublist = clean_sublist(target_dir,sublist)

for si = 1:length(sublist)
    oldname = sublist(si).name;
    name = strrep(oldname,'(','');
    name = strrep(name,')','');
    sublist(si).name = name;
    system(sprintf('mv ''%s'' %s',[target_dir filesep oldname],[target_dir filesep name]));
end;

return;


    
    