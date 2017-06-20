 clear all
 load database_kan_utf8_mono_db_50.mat
 rng default
 
% sentence_index={'902','907','910','914','915','919','920_1','927','955','959'};
% alpha_array=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0];
%sentence_index={'959','955','927','920_1','919','915','914','910','907','902'};
sentence_index={'sample'};
alpha_array=[0.5];

        G1 = G;
        clear G

for y=1:size(sentence_index,2)
    clear O T input_log_file fileID input_file_cell
    sentence_no=sentence_index{y};
%   sentence_no='one_word';
%   main_path=strcat('D:\New folder\New folder (4)\MATLAB\syllableviterbi\output_syl_10\t_',sentence_no,'\');
    main_path=strcat('output\mixed_viterbi\v1\t_',sentence_no,'\');
    mkdir(main_path);
%   main_path=strcat('C:\Users\shreyas\Documents\MATLAB\syllableviterbi\new\');
%   mkdir(main_path);
%   utf8_file=strcat('D:\New folder\New folder (4)\MATLAB\syllableviterbi\phrases_20\t_',sentence_no,'.txt');
    utf8_file=strcat('phrases_20\t_',sentence_no,'.txt');
%   utf8_file=strcat('C:\Users\shreyas\Documents\MATLAB\syllableviterbi\',sentence_no,'.txt');
%   fidleID = fopen(utf8_file,'r','n','UTF-8');
    fileID = fopen(utf8_file,'r','n','UTF-8');
    str1=fread(fileID);
    str2=dec2hex(str1);
    utf=cell(1,size(str2,1));
    for i=1:size(str2,1)
        utf{1,i}=strcat(str2(i,1),str2(i,2));
    end
    fclose(fileID);

    clear str1 str2 utf8_cell utf8_file fileID ans i

    O=kan_sen_to_utf_mono(utf,UTF8);
%   str = textscan(fidleID,'%f %d %s ','Delimiter', {'#'});
%   str1 = str{1,3}';
%   for i = 1: length(str1)-1
%       O{1,i} = str1{1,i+1};
%   end
%   fclose(fidleID);

    O=numbering(O,P);
    T=size(O,2);
    O=left_right_context(O);
    O=[num2cell(zeros(1,T));O];
    
    for z=1:size(alpha_array,2)
		time=zeros(1,T);
        alpha=alpha_array(z);
%       rng default
        main_path_alpha=strcat(main_path,'\alpha_',num2str(alpha),'\');
        mkdir(main_path_alpha);
		D=(10000*ones(T,S,Lmax));
        Si=(zeros(T,S,Lmax,2));
        fprintf('Sentence number = %s\tAlpha = %f\tStarting t = 1 of %d\n',sentence_no,alpha,T);
        t=1;
        tic;
        for i=1:S
                D(t,i,1)=(1-alpha)*Du(O(:,1),Unew{1,i}{1,1});
        end

        time(1)=toc;

        fprintf('Sentence number = %s\tAlpha = %f\tTime taken for t = 1 is %f\n',sentence_no,alpha,time(1));

        for t=2:T
            fprintf('Syllable Viterbi \nSentence number = %s\tAlpha = %f\tStarting t = %d of %d\n',sentence_no,alpha,t,T);
            tic;
            Ot=O(:,t);
            Dt1=D(t-1,:,:);
            Dt=10000*ones(S,Lmax);
            Sit=10000*ones(1,S,Lmax,2);
            parfor i=1:S
%             for i=1:S
                Unew1i11=Unew{1,i}{1,1};
                Dti=10000*ones(1,Lmax);  
                Siti=10000*ones(1,1,Lmax,2);
                for j=1:size(Unew{1,i},2) 
                    Unew1i1j=Unew{1,i}{1,j};
                     if j==1
                         sum=100*ones(1,S);
                         for k=1:S
%                                sum(1,k)=Dt1(1,k,size(Unew{1,k},2))+alpha*Dc(Unew{1,k}{1,size(Unew{1,k},2)},Unew1i11);
                               possible_cc=concat_cost_norm(Unew{1,k}{1,size(Unew{1,k},2)}{1,1},Unew1i11{1,1});
                               sum(1,k)=Dt1(1,k,size(Unew{1,k},2))+alpha*Dc_mfcc(Unew{1,k}{1,size(Unew{1,k},2)}{1,1},Unew1i11{1,1},possible_cc);                         
                         end
                         
                         %[mincost,index]=min(sum);
                          [mincost,index]=min_rand(sum);
                          
                         Dti(j)=mincost+(1-alpha)*Du(Ot,Unew1i11);
                        Siti(1,1,j,1)=index;
                        Siti(1,1,j,2)=size(Unew{1,index},2);
                     else
                        Dti(j)=Dt1(1,i,j-1)+(1-alpha)*Du(Ot,Unew1i1j);
                        Siti(1,1,j,1)=i;
                        Siti(1,1,j,2)=j-1;
                     end
                end 
                Dt(i,:)=Dti;
                Sit(1,i,:,:)=Siti;
            end
            clear mini mi min_temp min_index temp
            D(t,:,:)=Dt;
            Si(t,:,:,:)=Sit;
            time(t)=toc;
            fprintf('Sentence number = %s\tAlpha = %f\tTime taken for t = %d is %f\n',sentence_no,alpha,t,time(t));
        end
        time_avg=mean(time);

        %% optimum sequence index

        Q=zeros(2,T);
        Dmin=2000;
        for i=1:S
            for j=1:Lmax
                if D(T,i,j)<Dmin
                    Dmin=D(T,i,j);
                    Q(1,T)=i;
                    Q(2,T)=j;
                end
            end
        end

        for t=T-1:-1:1
            Q(1,t)=Si(t+1,Q(1,t+1),Q(2,t+1),1);
            Q(2,t)=Si(t+1,Q(1,t+1),Q(2,t+1),2);
        end

		for t=1:T
			O{1,t}=Unew{1,Q(1,t)}{1,Q(2,t)}{1,1};
        end
        
        O1 = O;
        
        %%
%         O=[O;num2cell(zeros(1,size(O,2)))];
%         for ii = 1:size(O,2)-1
%             if O{1,ii} == O{1,ii+1} - 1
%                 O{10,ii} = 0;
%             else
%                 O{10,ii} = 1;
%             end
%         end
%         clear ii;
%         O=[O;num2cell(zeros(1,size(O,2)))];
%         for ii = 2:size(O,2)
%             if O{1,ii} == O{1,ii-1} + 1
%                 O{11,ii} = 0;
%             else
%                 O{11,ii} = 1;
%             end
%         end
%         clear ii;
        
        %%
%         jj = 1;
%         for ii = 1:size(O,2) - 1
%             if O{10,ii} == 1
% %                er_syll(jj) = O{8,ii};
% %                el_syll(jj) = O{8,ii+1};
% %                distance_syll(jj) = dist(O{6,ii},O{6,ii+1}');
%                 el_syll(jj) = U{18,O{1,ii}};
%                 er_syll(jj) = U{17,(O{1,ii}+1)};
%                 distance_syll(jj) = dist(U{14,O{1,ii}},U{13,(O{1,ii}+1)}');
%                 jj = jj + 1;
%             end
%         end
%         clear ii jj;
%         eavg_syll = (er_syll + el_syll) / 2;
%         clear ii jj;
%         O_syll = O;
%         save('plot_syll.mat','el_syll','er_syll','distance_syll','eavg_syll', 'O_syll');
% %         figure;
% %         hold on;
% %         plot(distance_syll,el_syll,'*r');
% %         plot(distance_syll,er_syll,'*b');
% %         plot(distance_syll,eavg_syll,'*k');
% %         hold off;

        %% Du, Dc and Dmin for every alpha
        Du_opt=0;
        Dc_opt=0;
        for i=1:T
            if i~=1
%                 Dc_opt=Dc_opt+alpha*Dc(U(:,O{1,i-1}),U(:,O{1,i}));
                possible_cc=concat_cost_norm(U{1,O{1,i-1}},U{1,O{1,i}});
                Dc_opt=Dc_opt+alpha*Dc_mfcc(U{1,O{1,i-1}},U{1,O{1,i}},possible_cc);  
            end
%             if i==1
%                 Du_opt=Du_opt+Du(O(:,i),U(:,O{1,i}));
%             else
                Du_opt=Du_opt+(1-alpha)*Du(O(:,i),U(:,O{1,i}));
%             end
        end
        Dmin=Du_opt+Dc_opt;
        path1=strcat(main_path_alpha,'stage_1_Du_Dc_Dmin_',num2str(alpha),'.txt');
        fileID1 = fopen(path1,'w','n','UTF-8');
                fprintf(fileID1,'%f\t%f\t%f \r\n',Dmin,Du_opt,Dc_opt);
        fclose(fileID1);

        %% Du, Dc and Dmin for every alpha with no weights
        Du_opt=0;
        Dc_opt=0;
        for i=1:T
            if i~=1
%                 Dc_opt=Dc_opt+Dc(U(:,O{1,i-1}),U(:,O{1,i}));
                possible_cc=concat_cost_norm(U{1,O{1,i-1}},U{1,O{1,i}});
                Dc_opt=Dc_opt+Dc_mfcc(U{1,O{1,i-1}},U{1,O{1,i}},possible_cc);
                Dc_final(i) = Dc_mfcc(U{1,O{1,i-1}},U{1,O{1,i}},possible_cc);
            end
                Du_opt=Du_opt+Du(O(:,i),U(:,O{1,i}));
                Du_final(i) = Du(O(:,i),U(:,O{1,i}));
        end

        Dmin=Du_opt+Dc_opt;

        path1=strcat(main_path_alpha,'stage_1_Du_Dc_Dmin_no_weights',num2str(alpha),'.txt');
        fileID1 = fopen(path1,'w','n','UTF-8');
                fprintf(fileID1,'%f\t%f\t%f \r\n',Dmin,Du_opt,Dc_opt);
        fclose(fileID1);

        %% calculate the contiguous units
        C=contiguous(O(1,:));
        [Cr,Cc]=size(C);
        for i=1:Cc
            C(Cr+1,i)=nnz(C(:,i));
        end

        Countc=max(C(Cr+1,:));
        Count=zeros(1,Countc);
        for i=1:Cc
            Count(C(Cr+1,i))=Count(C(Cr+1,i))+1;
        end


        %% units selected
        path1=strcat(main_path_alpha,'stage_1_units_alpha_',num2str(alpha),'.txt');
        fileID1 = fopen(path1,'w','n');
        for i = 1:T
            fprintf(fileID1,'unit number = %d\tUnit label = %s\tInput unit\tlabel = %s\tSyllable Number = %d\tMonophone in Syllable number = %d\r\n',O{1,i},U{2,O{1,i}},char(O{2,i}),Q(1,i),Q(2,i));
		end
		fclose(fileID1);
		
		%% Number of syllable count
		path1=strcat(main_path_alpha,'stage_1_Count_syllables_',num2str(alpha),'.txt');
		fileID1=fopen(path1,'w','n');
		syl_count=0;
		for i=1:T
			if Q(2,i)==1
				syl_count=syl_count+1;
			end
		end
		fprintf(fileID1,'Number of syllables = %d',syl_count);
		fclose(fileID1);
        
		%% Number of substitution errors
		sub_error=0;
		for i=1:T
			if (~(strcmp(U{2,O{1,i}},char(O{2,i}))))
				sub_error=sub_error+1;
			end
		end
		path1=strcat(main_path_alpha,'stage_1_sub_errors_alpha_',num2str(alpha),'.txt');
		fileID1 = fopen(path1,'w','n');
		fprintf(fileID1,'%d\t%d',sub_error,T);
		fclose(fileID1);

        %% printing sorted contiguous units

        path1=strcat(main_path_alpha,'stage_1_contiguous_alpha_',num2str(alpha),'.txt');
        fileID1 = fopen(path1,'w','n','UTF-8');
         for j=1:Cc 
             for i = 1:Cr+1
                fprintf(fileID1,'%d \r\n',C(i,j));
             end
             fprintf(fileID1,'\n');
         end
        fclose(fileID1);

        %% count of contiguous units

        path1=strcat(main_path_alpha,'stage_1_count_alpha_',num2str(alpha),'.txt');
        fileID1 = fopen(path1,'w','n','UTF-8');
         for i = 1:size(Count,2)
                fprintf(fileID1,'%d \r\n',Count(1,i));
         end
        fclose(fileID1);    

        %% avg length of contigous units
        conti_sum=0;
        s=0;
        for i=1:size(Count,2)
            conti_sum=conti_sum+(Count(1,i)*i);
            s=s+Count(1,i);
        end
        Cavg=conti_sum/s;

        path1=strcat(main_path_alpha,'stage_1_Cavg_',num2str(alpha),'.txt');
        fileID1 = fopen(path1,'w','n','UTF-8');
                fprintf(fileID1,'%d \r\n',Cavg);
        fclose(fileID1);

        %%
        path1=strcat(main_path_alpha,'stage_1_Timeavg_',num2str(alpha),'.txt');
        fileID1 = fopen(path1,'w','n','UTF-8');
                fprintf(fileID1,'%d \r\n',time_avg);
        fclose(fileID1); 

        %% for every alpha
        % Concatenate the units selected.
        [wavefile,Fs]=waveform(U,O);
        path1=strcat(main_path_alpha,'stage_1_wave_alpha_',num2str(alpha),'.wav');
        audiowrite(path1,wavefile,Fs);     
        
       %%
        O = [O; num2cell(Dc_final); num2cell(Du_final)];
        final = gen_log(U,O);
        path1=strcat(main_path_alpha,'stage_1_log_',num2str(alpha),'.txt');
        fileID1 = fopen(path1,'w','n','UTF-8');
                fprintf(fileID1,'%d \r\n',time_avg);
        fclose(fileID1); 
        writetable(cell2table(final','VariableNames',{'t', 'Input_UTF8', 'Input_Eng_Phn', 'Output_Unit_Number', 'Output_UTF8', 'Output_Eng_Phn', 'Start_time', 'End_time', 'Monophone_Count_in_contiguous_Units', 'Contiguous_Unit_Count', 'Concat_Cost', 'Unit_Cost'}),path1,'Delimiter','tab');
        
        path1=strcat(main_path_alpha,'stage_1_workspace',num2str(alpha),'.mat');
        save(path1);
        clear Du_final Dc_final O final
        
        %%
        clear ans C Cavg Cc conti_sum Count Countc Cr D Dc_opt Dmin Dt Dt1 Du_opt Index_start Index_end fileID1 
        clear Fs i j k path Q s Si Ot path1 a percentage_of_monophones Sit sub_error syl_count mono_count 
        clear sum t time time_avg wavefile
        
        %%
        [O,G] = stage_2(O1,G1,U);
        
        r=zeros(3,T);
        for i=1:T
            [k,r(1,i)]=find(G(1,:)==O{3,i});
            r(2,i)=G(1,r(1,i));
            r(3,i)=G(2,r(1,i));
        end
%       alpha=0.4;
        D=ones(T,max(r(3,:)));
        D=100*D;
        Si=zeros(T,max(r(3,:)));
        time=zeros(1,T);
        fprintf('Sentence number = %s\tAlpha = %f\tStarting t = 1\n',sentence_no,alpha);
        t=1;
        tic;
        for i=1:r(3,1)
            D(1,i)=Du(O(:,1),U(:,G(i+2,r(1,1))));
        end
        time(1)=toc;
        fprintf('Sentence number = %s\tAlpha = %f\tTime taken for t = %d is %f\n',sentence_no,alpha,t,time(t));
        
        for t=2:T
            fprintf('Monophone Viterbi \nSentence number = %s\tAlpha = %f\tStarting t = %d\n',sentence_no,alpha,t);
            tic;
            
            Dt1 = D(t-1,:);
            %     for i=1:r(3,t)
            parfor i=1:r(3,t)
                sum=zeros(1,r(3,t-1));
                gi=G(i+2,r(1,t));
                for j=1:r(3,t-1)
                    gj=G(j+2,r(1,t-1));
                    %             sum(j)=D(t-1,j)+alpha*Dc(U(:,gj),U(:,gi));
                    possible_cc=concat_cost_norm(U{1,gj},U{1,gi});
                    sum(j)=Dt1(1,j)+alpha*Dc_mfcc(U{1,gj},U{1,gi},possible_cc);
                    %             [mincost,index]=min(sum);
                    
                end
                
                [mincost,index]=min_rand(sum);
                D(t,i)=mincost+(1-alpha)*Du(O(:,t),U(:,gi));
                Si(t,i)=index;
                
            end
            %     clear mini mi min_temp min_index temp
            clear possible_cc
            time(t)=toc;
            fprintf('Sentence number = %s\tAlpha = %f\tTime taken for t = %d of T = %d is %f\n',sentence_no,alpha,t,T,time(t));
        end
        time_avg=mean(time);
        clear mincost index gi gj
        
        %% optimum sequence index
        
        % 		Q=zeros(2,T);
        % 		Dmin=2000;
        % 		for i=1:Us
        % 			for j=1:Lmax
        % 				if D(T,i,j)<Dmin
        % 					Dmin=D(T,i,j);
        % 					Q(1,T)=i;
        % 					Q(2,T)=j;
        % 				end
        % 			end
        % 		end
        %
        %         for t=T-1:-1:1
        %             Q(1,t)=Si(t+1,Q(1,t+1),Q(2,t+1),1);
        % 			Q(2,t)=Si(t+1,Q(1,t+1),Q(2,t+1),2);
        %         end
        %
        %         for t=1:T
        %             O{1,t}=Unew{1,Q(1,t)}{1,Q(2,t)}{1,1};
        %         end
        
        %  [mincost,index]=min(D(T,:));
        [mincost,index]=min_rand(D(T,:));
        Dmin = mincost;
        Q=zeros(2,T);
        Q(1,T)=index;
        Q(2,T)=G(index+2,r(1,T));
        
        for t=T-1:-1:1
            Q(1,t)=Si(t+1,Q(1,t+1));
            Q(2,t)=G((Q(1,t)+2),r(1,t));
        end
        
        O(1,:)=num2cell(Q(2,:));
        %toc;
        clear mincost index
        
        %    [mincost,index]=min(D(T,:));
        %             Q=zeros(2,T);
        %             Q(1,T)=index;
        %             Q(2,T)=G(index+2,r(1,T));
        %
        %             for t=T-1:-1:1
        %                 Q(1,t)=Si(t+1,Q(1,t+1));
        %                  Q(2,t)=G((Q(1,t)+2),r(1,t));
        %             end
        %
        %             O(1,:)=num2cell(Q(2,:));
        %             for t=1:T
        %
        %             O{6,t} = U{13,O{1,t}};
        %             O{7,t} = U{14,O{1,t}};
        %             O{8,t} = U{15,O{1,t}};
        %             O{9,t} = U{16,O{1,t}};
        %         end
        % %toc;
        % clear mincost index
        %         %%
        %         O=[O;num2cell(zeros(1,size(O,2)))];
        %         for ii = 1:size(O,2)-1
        %             if O{1,ii} == O{1,ii+1} - 1
        %                 O{10,ii} = 0;
        %             else
        %                 O{10,ii} = 1;
        %             end
        %         end
        %         clear ii;
        %         O=[O;num2cell(zeros(1,size(O,2)))];
        %         for ii = 2:size(O,2)
        %             if O{1,ii} == O{1,ii-1} + 1
        %                 O{11,ii} = 0;
        %             else
        %                 O{11,ii} = 1;
        %             end
        %         end
        %         clear ii;
        %
        %         %%
        %         jj = 1;
        %         for ii = 1:size(O,2) - 1
        %             if O{10,ii} == 1
        %                 el_mono(jj) = U{18,O{1,ii}};
        %                 er_mono(jj) = U{17,(O{1,ii}+1)};
        %                 distance_mono(jj) = dist(U{14,O{1,ii}},U{13,(O{1,ii}+1)}');
        %                 jj = jj + 1;
        %             end
        %         end
        %         eavg_mono = (er_mono+el_mono)/2;
        %         clear ii jj;
        % %         for i =
        %          O_mono = O;
        % %         figure;
        % %         hold on;
        % %         plot(eu_dist_mono,el_mono,'*r');
        % %         plot(eu_dist_mono,er_mono,'*b');
        % %         plot(eu_dist_mono,eavg_mono,'*k');
        % %         hold off;
        %         save('plot_mono.mat','el_mono','er_mono','distance_mono','eavg_mono','O_mono');
        
        
        %% Du, Dc and Dmin for every alpha
        Du_opt=0;
        Dc_opt=0;
        for i=1:T
            if i~=1
                possible_cc=concat_cost_norm(U{1,O{1,i-1}},U{1,O{1,i}});
                Dc_opt=Dc_opt+alpha*Dc_mfcc(U{1,O{1,i-1}},U{1,O{1,i}},possible_cc);
                %                 Dc_opt=Dc_opt+alpha*Dc(U(:,O{1,i-1}),U(:,O{1,i}));
            end
            if i==1
                Du_opt=Du_opt+Du(O(:,i),U(:,O{1,i}));
            else
                Du_opt=Du_opt+(1-alpha)*Du(O(:,i),U(:,O{1,i}));
            end
        end
        path1=strcat(main_path_alpha,'Du_Dc_Dmin_',num2str(alpha),'.txt');
        fileID1 = fopen(path1,'w','n','UTF-8');
        fprintf(fileID1,'%f\t%f\t%f \r\n',Dmin,Du_opt,Dc_opt);
        %                 fprintf(fileID1,'%f\t%f \r\n',Du_opt,Dc_opt);
        fclose(fileID1);
        
        %% Du, Dc and Dmin for every alpha with no weights
        Du_opt=0;
        Dc_opt=0;
        for i=1:T
            if i~=1
                possible_cc=concat_cost_norm(U{1,O{1,i-1}},U{1,O{1,i}});
                Dc_opt=Dc_opt+alpha*Dc_mfcc(U{1,O{1,i-1}},U{1,O{1,i}},possible_cc);
                Dc_final(i) = Dc_mfcc(U{1,O{1,i-1}},U{1,O{1,i}},possible_cc);
                %                 Dc_opt=Dc_opt+Dc(U(:,O{1,i-1}),U(:,O{1,i}));
                %                 Dc_final(i) = Dc(U(:,O{1,i-1}),U(:,O{1,i}));
            end
            Du_opt=Du_opt+Du(O(:,i),U(:,O{1,i}));
            Du_final(i) = Du(O(:,i),U(:,O{1,i}));
        end
        
        Dmin=Du_opt+Dc_opt;
        
        path1=strcat(main_path_alpha,'Du_Dc_Dmin_no_weights',num2str(alpha),'.txt');
        fileID1 = fopen(path1,'w','n','UTF-8');
        fprintf(fileID1,'%f\t%f\t%f \r\n',Dmin,Du_opt,Dc_opt);
        fclose(fileID1);
        
        %% calculate the contiguous units
        C=contiguous(O(1,:));
        [Cr,Cc]=size(C);
        for i=1:Cc
            C(Cr+1,i)=nnz(C(:,i));
        end
        
        Countc=max(C(Cr+1,:));
        Count=zeros(1,Countc);
        for i=1:Cc
            Count(C(Cr+1,i))=Count(C(Cr+1,i))+1;
        end
        
        
        %% units selected
        path1=strcat(main_path_alpha,'units_alpha_',num2str(alpha),'.txt');
        fileID1 = fopen(path1,'w','n');
        for i = 1:T
            fprintf(fileID1,'unit number = %d\tUnit label = %s\tInput unit\tlabel = %s\tSyllable Number = %d\tMonophone in Syllable number = %d\r\n',O{1,i},U{2,O{1,i}},char(O{2,i}),Q(1,i),Q(2,i));
        end
        fclose(fileID1);
        
        %% Number of syllable count
        path1=strcat(main_path_alpha,'Count_syllables_',num2str(alpha),'.txt');
        fileID1=fopen(path1,'w','n');
        syl_count=0;
        for i=1:T
            if Q(2,i)==1
                syl_count=syl_count+1;
            end
        end
        fprintf(fileID1,'Number of syllables = %d',syl_count);
        fclose(fileID1);
        
        %% Number of substitution errors
        sub_error=0;
        for i=1:T
            if (~(strcmp(U{2,O{1,i}},char(O{2,i}))))
                sub_error=sub_error+1;
            end
        end
        path1=strcat(main_path_alpha,'sub_errors_alpha_',num2str(alpha),'.txt');
        fileID1 = fopen(path1,'w','n');
        fprintf(fileID1,'%d\t%d',sub_error,T);
        fclose(fileID1);
        
        %% printing sorted contiguous units
        
        path1=strcat(main_path_alpha,'contiguous_alpha_',num2str(alpha),'.txt');
        fileID1 = fopen(path1,'w','n','UTF-8');
        for j=1:Cc
            for i = 1:Cr+1
                fprintf(fileID1,'%d \r\n',C(i,j));
            end
            fprintf(fileID1,'\n');
        end
        fclose(fileID1);
        
        %% count of contiguous units
        
        path1=strcat(main_path_alpha,'count_alpha_',num2str(alpha),'.txt');
        fileID1 = fopen(path1,'w','n','UTF-8');
        for i = 1:size(Count,2)
            fprintf(fileID1,'%d \r\n',Count(1,i));
        end
        fclose(fileID1);
        
        %% avg length of contigous units
        conti_sum=0;
        s=0;
        for i=1:size(Count,2)
            conti_sum=conti_sum+(Count(1,i)*i);
            s=s+Count(1,i);
        end
        Cavg=conti_sum/s;
        
        path1=strcat(main_path_alpha,'Cavg_',num2str(alpha),'.txt');
        fileID1 = fopen(path1,'w','n','UTF-8');
        fprintf(fileID1,'%d \r\n',Cavg);
        fclose(fileID1);
        
        %%
        path1=strcat(main_path_alpha,'Timeavg_',num2str(alpha),'.txt');
        fileID1 = fopen(path1,'w','n','UTF-8');
        fprintf(fileID1,'%d \r\n',time_avg);
        fclose(fileID1);
        
        %% for every alpha
        %Concatenate the units selected.
        [wavefile,Fs]=waveform(U,O);
        path1=strcat(main_path_alpha,'wave_alpha_',num2str(alpha),'.wav');
        audiowrite(path1,wavefile,Fs);
        
        %%
        O = [O; num2cell(Dc_final); num2cell(Du_final)];
        final = gen_log(U,O);
        path1=strcat(main_path_alpha,'log_',num2str(alpha),'.txt');
        fileID1 = fopen(path1,'w','n','UTF-8');
        fprintf(fileID1,'%d \r\n',time_avg);
        fclose(fileID1);
        writetable(cell2table(final','VariableNames',{'t', 'Input_UTF8', 'Input_Eng_Phn', 'Output_Unit_Number', 'Output_UTF8', 'Output_Eng_Phn', 'Start_time', 'End_time', 'Monophone_Count_in_contiguous_Units', 'Contiguous_Unit_Count', 'Concat_Cost', 'Unit_Cost'}),path1,'Delimiter','tab');
        
        clear O Dc_final Du_final final
        
        
        
    end
	clear alpha ans C Cavg Cc conti_sum Count Countc Cr D Dc_opt Dmin Dt Dt1 Du_opt Index_start Index_end fileID1 
	clear Fs i j k path Q s Si Ot path1 a percentage_of_monophones Sit sub_error syl_count mono_count Us 
	clear sum t time time_avg wavefile
 end