% Re-render the five-panel GSE174554 expression validation figure.
% This script consumes only frozen Source Data CSV files.
root = fileparts(fileparts(mfilename('fullpath')));
s = fullfile(root, 'source_data');
pairs = readtable(fullfile(s, 'FigExpr_A_processed_pairs.csv'));
honest = readtable(fullfile(s, 'FigExpr_B_honest_candidate_summary.csv'));
lopo = readtable(fullfile(s, 'FigExpr_C_lopo_seed_summary.csv'));
stab = readtable(fullfile(s, 'FigExpr_D_bootstrap_stability.csv'));
del = readtable(fullfile(s, 'FigExpr_E_delete_one.csv'));
lopo = lopo(lopo.seed == 20260903,:);
stab = stab(stab.seed == 20260903,:);
f = figure('Color','w','Position',[100 100 1550 335]);
tiledlayout(f,1,5,'Padding','compact','TileSpacing','compact');
nexttile; bar([30 height(pairs) sum(strcmp(pairs.analysis_partition,'development')) sum(strcmp(pairs.analysis_partition,'confirmation'))]); title('A  Auditable cohort'); ylabel('Physical patients'); xticklabels({'Frozen','Effective','Development','Confirmation'}); xtickangle(35);
nexttile; hold on; y=(1:height(honest))'; for i=1:height(honest), plot([honest.utility_lcb(i) honest.utility_estimate(i)],[i i],'-','LineWidth',2); end; scatter(honest.utility_estimate,y,20,'filled'); xline(0,'--k'); yticks(y); yticklabels(honest.candidate); set(gca,'YDir','reverse'); title('B  Honest confirmation');
nexttile; y=(1:height(lopo))'; scatter(lopo.candidate_mae,y,20,'filled'); hold on; scatter(lopo.reference_mae,y,20); yticks(y); yticklabels(lopo.candidate); set(gca,'YDir','reverse'); title('C  Dependent LOPO'); xlabel('Patient-equal MAE');
nexttile; hold on; names=unique(stab.candidate,'stable'); for i=1:numel(names), z=stab(strcmp(stab.candidate,names{i}),:); errorbar(1:height(z),z.PUC,z.PUC-z.PUC_lo,z.PUC_hi-z.PUC,'-o'); end; yline(0,'--k'); xticks(1:4); xticklabels({'250','500','1000','2000'}); title('D  Bootstrap stability'); xlabel('Bootstrap B');
nexttile; names=unique(del.candidate,'stable'); x=[]; g={}; for i=1:numel(names), z=del.utility_lcb(strcmp(del.candidate,names{i}) & strcmp(del.status,'SUCCESS')); x=[x;z]; g=[g;repmat(names(i),numel(z),1)]; end; boxplot(x,g); yline(0,'--k'); title('E  Full-pipeline deletion'); xtickangle(35);
exportgraphics(f,fullfile(root,'figures','gse174554_expression_validation_matlab.pdf'),'ContentType','vector');
