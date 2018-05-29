function generate_brianseg_pdf_xiong_light(rawfile, segfn, pdfname, cmap_orig_vals,...
    temp_dir,proj_name, subj_name, expr_name, scan_name, spider_name, filenum, totalnum)

% GENERATE_ABDOMENSEG_PDF - Generates the summary PDF for abdominal organ
%                           segmentation
%
% Input: subFile - the raw input file (intensity image filename -- .nii.gz)
%        segfn - the estimated segmentation -- .nii.gz
%        pdfname - the final pdf filename
%        cmap - the colormap of abdominal organs
%        temp_dir - the directory to store temporary output
%        proj_name - the name of the project
%        subj_name - the name of the subject
%        expr_name - the name of the experiment
%        scan_name - the name of the scan
%        spider_name - the name of the spider
%		 filenum : current scan #
%		 totalnum : total scan #
% Output: None.

% load colormap


% load images
RawNii=load_untouch_nii_gz(rawfile);
RawNii.img=double(RawNii.img+RawNii.hdr.dime.scl_inter);
SegNii=load_untouch_nii_gz(segfn);


% set some label information
ll = unique(SegNii.img);
ll = ll(2:end);
la = ones(size(ll));
la(ll >= 100) = 0;
la(ll >= 38 & ll <= 45) = 0;
la(ll >= 71 & ll <= 73) = 0;
la(ll == 35) = 0;
ll2 = ll(la > 0);

% set the default options
default_opts.fignum = 99;
default_opts.labels = ll;
default_opts.labelcolors = cmap_orig_vals(2:end, :);
default_opts.material = 'dull';
default_opts.slicealpha = 0.9;
default_opts.cr_buffer = -1;
default_opts.ilim = get_optimal_ilim(RawNii.img, SegNii.img);
default_opts.xslices = round(SegNii.hdr.dime.dim(2)/2)+10;
default_opts.yslices = round(SegNii.hdr.dime.dim(3)/2);
default_opts.zslices = round(SegNii.hdr.dime.dim(4)/2);

                      
                  
%  axial axes1
im_axi = plot_segmentation_overlay(permute(RawNii.img, [2 1 3]), ...
                          permute(SegNii.img, [2 1 3]), ...
                          0.0, cmap_orig_vals, default_opts.ilim, ...
                          default_opts.zslices);
                      
% coronal axes18
im_cor = plot_segmentation_overlay(flipdim(permute(RawNii.img, [3 1 2]), 1), ...
                          flipdim(permute(SegNii.img, [3 1 2]), 1), ...
                          0.0, cmap_orig_vals, default_opts.ilim, ...
                          default_opts.yslices);            
                      
% saggital axes19
im_sag = plot_segmentation_overlay(flipdim(permute(RawNii.img, [3 2 1]), 1), ...
                          flipdim(permute(SegNii.img, [3 2 1]), 1), ...
                          0.0, cmap_orig_vals, default_opts.ilim, ...
                          default_opts.yslices);                 
                      
% axial segmentation axes21    
im_axi_seg = plot_segmentation_overlay(permute(RawNii.img, [2 1 3]), ...
                          permute(SegNii.img, [2 1 3]), ...
                          0.6, cmap_orig_vals, default_opts.ilim, ...
                          default_opts.zslices);                      

% coronal segmentation axes22
im_cor_seg = plot_segmentation_overlay(flipdim(permute(RawNii.img, [3 1 2]), 1), ...
                          flipdim(permute(SegNii.img, [3 1 2]), 1), ...
                          0.6, cmap_orig_vals, default_opts.ilim, ...
                          default_opts.yslices);                      

% saggital segmentation axes23  
im_sag_seg = plot_segmentation_overlay(flipdim(permute(RawNii.img, [3 2 1]), 1), ...
                          flipdim(permute(SegNii.img, [3 2 1]), 1), ...
                          0.6, cmap_orig_vals, default_opts.ilim, ...
                          default_opts.xslices);  

% % cortical segmentation axes24
% opts = default_opts;
% opts.azimuth = 90;
% opts.elevation = 90;
% render_3D_labels(RawNii.img, SegNii.img, opts);
% set(gcf, 'color', 'w');
% set(gca, 'units', 'pixels');
% fr1 = getframe(gcf, get(gca, 'Position'));
% cr1 = determine_cropping_region(min(fr1.cdata, [], 3)<255, 0);
% 
% % cortical segmentation axes25
% opts = default_opts;
% opts.azimuth = 90;
% opts.elevation = 0;
% render_3D_labels(RawNii.img, SegNii.img, opts);
% set(gcf, 'color', 'w');
% set(gca, 'units', 'pixels');
% fr2 = getframe(gcf, get(gca, 'Position'));
% cr2 = determine_cropping_region(min(fr2.cdata, [], 3)<255, 0);
% 
% % cortical segmentation axes26
% opts = default_opts;
% opts.azimuth = 0;
% opts.elevation = 0;
% render_3D_labels(RawNii.img, SegNii.img, opts);
% set(gcf, 'color', 'w');
% set(gca, 'units', 'pixels');
% fr3 = getframe(gcf, get(gca, 'Position'));
% cr3 = determine_cropping_region(min(fr3.cdata, [], 3)<255, 0);
% 
% % mid-brain segmentation axes27
% opts = default_opts;
% opts.labelalphas = la;
% opts.azimuth = 90;
% opts.elevation = 90;
% render_3D_labels(RawNii.img, SegNii.img, opts);
% set(gcf, 'color', 'w');
% set(gca, 'units', 'pixels');
% fr4 = getframe(gcf, get(gca, 'Position'));
% cr4 = determine_cropping_region(min(fr4.cdata, [], 3)<255, 0);
% 
% % mid-brain segmentation axes28
% opts = default_opts;
% opts.labelalphas = la;
% opts.azimuth = 90;
% opts.elevation = 0;
% render_3D_labels(RawNii.img, SegNii.img, opts);
% set(gcf, 'color', 'w');
% set(gca, 'units', 'pixels');
% fr5 = getframe(gcf, get(gca, 'Position'));
% cr5 = determine_cropping_region(min(fr5.cdata, [], 3)<255, 0);
% 
% % mid-brain segmentation axes29
% opts = default_opts;
% opts.labelalphas = la;
% opts.azimuth = 0;
% opts.elevation = 0;
% render_3D_labels(RawNii.img, SegNii.img, opts);
% set(gcf, 'color', 'w');
% set(gca, 'units', 'pixels');
% fr6 = getframe(gcf, get(gca, 'Position'));
% cr6 = determine_cropping_region(min(fr6.cdata, [], 3)<255, 0);


%--------------------------------Arrange images---------------------------%
fig_main = figure(filenum);
set(fig_main,'Units','Inches','Position',[0 0 8.5 11],'Color','w');
a_main=axes('position',[0.05 0.05 0.9 0.90]);axis(a_main,'off');
title(a_main,'SLANT Deep Whole Brain Segmentation Result','FontSize',20);
a_raw=axes('position',[0.1 0.4 0.8,0.9]);axis(a_raw,'off');
text(-0.05,0.5,'Raw Image','horizontalalignment','center',...
    'parent',a_raw,'rotation',90,'fontsize',12);

wim_axi=0.23;
wim_cor=0.2;
wim_sag=0.255;
him=0.333;
wband=(0.4-wim_axi)/2;
hband=(0.3-him);hband_bottom=1/4*hband;

% begin raw image 
r_axi=axes('position',[wband,0.4+hband+hband_bottom+him,wim_axi,him]);
imshow(imrotate(im_axi,-90),'parent',r_axi);

r_cor=axes('position',...
         [wband+wim_axi+0.05,0.4+hband+hband_bottom+him,wim_cor,him]);
imshow(im_cor,'parent',r_cor);

r_sag = axes('position',...
 [wband+wim_axi+wim_cor+0.1,0.4+hband+hband_bottom+him,wim_sag,him]);
imshow(im_sag,'parent',r_sag);


% begin segmentation
a_seg=axes('position',[0.1 0.21 0.8,0.9]);axis(a_seg,'off');
text(-0.05,0.5,sprintf('Slicewise\nSegmentation'),'horizontalalignment','center',...
    'parent',a_seg,'rotation',90,'fontsize',12);

r_axi_seg = axes('position',[wband,0.4+hband+hband_bottom+0.4*him,wim_axi,him]);
imshow(imrotate(im_axi_seg,-90),'parent',r_axi_seg);

r_cor_seg = axes('position',...
         [wband+wim_axi+0.05,0.4+hband+hband_bottom+0.4*him,wim_cor,him]);
imshow(im_cor_seg,'parent',r_cor_seg);

r_sag_seg = axes('position',...
[wband+wim_axi+wim_cor+0.1,0.4+hband+hband_bottom+0.4*him,wim_sag,him]);
imshow(im_sag_seg,'parent',r_sag_seg);

% begin cortical segmentation
% a_cor_seg=axes('position',[0.1 0.02 0.8,0.9]);axis(a_cor_seg,'off');
% text(-0.05,0.5,sprintf('Cortical\nSegmentation'),'horizontalalignment','center',...
%     'parent',a_cor_seg,'rotation',90,'fontsize',12);
% 
% cor_axi_seg = axes('position',[wband,0.4+hband+hband_bottom-0.18*him,wim_axi,him]);
% tem1 = fr1.cdata(cr1(1):cr1(2), cr1(3):cr1(4), :);
% imshow(imrotate(tem1,-90),'parent',cor_axi_seg);
% 
% cor_cor_seg = axes('position',...
%          [wband+wim_axi+0.05,0.4+hband+hband_bottom-0.18*him,wim_cor,him]);
% tem2 = fr2.cdata(cr2(1):cr2(2), cr2(3):cr2(4), :);
% imshow(tem2,'parent',cor_cor_seg); 
% 
% cor_sag_seg = axes('position',...
% [wband+wim_axi+wim_cor+0.1,0.4+hband+hband_bottom-0.18*him,wim_sag,him]);
% tem3 = fr3.cdata(cr3(1):cr3(2), cr3(3):cr3(4), :);
% imshow(tem3,'parent',cor_sag_seg);
% 
% begin mid-brain segmentation
% a_cor_seg=axes('position',[0.1 0.03 0.8,0.5]);axis(a_cor_seg,'off');
% text(-0.05,0.5,sprintf('Mid-Brain\nSegmentation'),'horizontalalignment','center',...
%     'parent',a_cor_seg,'rotation',90,'fontsize',12);
% 
% mid_axi_seg = axes('position',[wband,0.4+hband+hband_bottom-0.75*him,wim_axi,him]);
% tem4 = fr4.cdata(cr4(1):cr4(2), cr4(3):cr4(4), :);
% imshow(imrotate(tem4,-90),'parent',mid_axi_seg);
% 
% mid_cor_seg = axes('position',...
%          [wband+wim_axi+0.05,0.4+hband+hband_bottom-0.75*him,wim_cor,him]);
% tem5 = fr5.cdata(cr5(1):cr5(2), cr5(3):cr5(4), :); 
% imshow(tem5,'parent',mid_cor_seg);
% 
% mid_sag_seg = axes('position',...
%          [wband+wim_axi+wim_cor+0.1,0.4+hband+hband_bottom-0.75*him,wim_sag,him]);
% tem6 = fr6.cdata(cr6(1):cr6(2), cr6(3):cr6(4), :);
% imshow(tem6,'parent',mid_sag_seg);

% Information Panel
a_info=axes('position',[0.1 0.4 0.8,0.14]);axis(a_info,'off');
text(-0.05,0.5,'INFO','horizontalalignment','center',...
    'parent',a_info,'rotation',90,'fontsize',12);
rectangle('position',[0 0 1 1],...
    'edgecolor',[0.3 0.3 0.3],'parent',a_info);
[~,rundate]=system('date');

indent=0.02;
project_label=text(indent,0.90,...
    'Project: ',...
    'fontsize',9,'fontweight','bold',...
    'interpreter','none');
et=get(project_label,'extent');
ns=et(1)+et(3)+indent;
text(ns,0.90,sprintf('%s',proj_name),'fontsize',9,'interpreter','none');

subject_label=text(indent,0.70,...
    'Subject: ',...
    'fontsize',9,'fontweight','bold',...
    'interpreter','none');
et=get(subject_label,'extent');    
ns=et(1)+et(3)+1*indent;
text(ns,0.70,sprintf('%s',subj_name),'fontsize',9,'interpreter','none');

experiment_label=text(indent,0.50,...
    'Experiment: ',...
    'fontsize',9,'fontweight','bold',...
    'interpreter','none');
et=get(experiment_label,'extent');    
ns=et(1)+et(3)+1*indent;
text(ns,0.5,sprintf('%s',expr_name),'fontsize',9,'interpreter','none');

contact_label=text(indent,0.30,...
    'Contact: ',...
    'fontsize',9,'fontweight','bold',...
    'interpreter','none');
et=get(contact_label,'extent');    
ns=et(1)+et(3)+1*indent;
contact = 'yuankai.huo@vanderbilt.edu';
text(ns,0.3,contact,'fontsize',9,'interpreter','none');

t_rundate_label=text(indent,0.10,...
    'Date of run: ',...
    'fontsize',9,'fontweight','bold',...
    'interpreter','none');
et=get(t_rundate_label,'extent');
ns=et(1)+et(3)+1*indent;
text(ns,0.10,...
    rundate(1:end-1),...
    'fontsize',9,...
    'interpreter','none');

t_versionDate_label=text(indent+0.6,0.10,...
    'Version Date: ',...
    'fontsize',9,'fontweight','bold',...
    'interpreter','none');
et=get(t_versionDate_label,'extent');
ns=et(1)+et(3)+1*indent;
versionDate = 'Wed May 23 2018';
text(ns,0.10,...
    versionDate,...
    'fontsize',9,...
    'interpreter','none');

url_label=text(indent,0.20,...
    'url: ',...
    'fontsize',9,'fontweight','bold',...
    'interpreter','none');
et=get(url_label,'extent');
ns=et(1)+et(3)+1*indent;
url = 'https://github.com/MASILab/SLANT_brain_seg';
text(ns,0.20,...
    url,...
    'fontsize',9,...
    'interpreter','none');

temp_ps = [temp_dir, '/temp.ps'];
print('-dpsc2','-r400', temp_ps, fig_main);
cmmd = ['ps2pdf ' temp_ps ' ' pdfname];
[status,msg]=system(cmmd);
if status~=0
    fprintf('\n Could not cleanly create pdf file from ps.\n');
    disp(msg);
end

end





