function nice_plot( MAT, name, offset)
%NICE_PLOT Summary of this function goes here
%   Detailed explanation goes here

switch name
    case 'D'
        md = MAT;
        figure(1)
        % figure('Position',[10,10,500,800],'Name','1');
        clf
        m = 16;
        sc = 0.06;
        for p = 1:m
        %     subplot(m, 1, p)
            if mod(p,2) == 0
                subplot('Position',[0.1,0.92-(p - 1)*sc,0.34,0.085])
            else
                subplot('Position',[0.5,0.92-p*sc,0.34,0.085])
            end
            stem(squeeze(md(:, p + offset)))
            xl = get(gca,'XTickLabel');
            set(gca,'XTickLabel','')
            if mod(p, 2) ~= 0
                set(gca, 'yaxislocation', 'right')
            end
            handle=title(['Element ', num2str(p + offset)]);
            if mod(p,2) == 0
                set(handle,'Position',[-6, 0, 0]);
            else
                set(handle,'Position',[40, 0, 0]);
            end
        end
        set(gca,'XTickLabel',xl)

    case 'S'
        ms = MAT;
        figure(1)
        % figure('Position',[10,10,500,800],'Name','1');
        clf
        m = 16;
        sc = 0.06;
        for p = 1:m
        %     subplot(m, 1, p)
            if mod(p,2) == 0
                subplot('Position',[0.1,0.92-(p - 1)*sc,0.34,0.085])
            else
                subplot('Position',[0.5,0.92-p*sc,0.34,0.085])
            end
            stem(squeeze(ms(:, p + offset)))
            xl = get(gca,'XTickLabel');
            set(gca,'XTickLabel','')
            if mod(p, 2) ~= 0
                set(gca, 'yaxislocation', 'right')
            end
            handle=title(['Element ', num2str(p + offset)]);
            if mod(p,2) == 0
                set(handle,'Position',[-6, 0, 0]);
            else
                set(handle,'Position',[75, 0, 0]);
            end
        end
        set(gca,'XTickLabel',xl)
    case 'Y'
        p1_data = MAT;
        figure(1)
        clf
        m = 15;
        sc = 0.063;
        for p = 1:m
        %     subplot(m, 1, p)
            subplot('Position',[0.1,0.99-p*sc,0.75,0.06])
            plot(squeeze(p1_data(1,p + offset,1:500)))
            xl = get(gca,'XTickLabel');
            set(gca,'XTickLabel','')
            if mod(p, 2) == 0
                set(gca, 'yaxislocation', 'right')
            end
            handle=title(['Channel ', num2str(p + offset)]);
            set(handle,'Position',[-30, 0, 0]);
        end
        set(gca,'XTickLabel',xl)
        
end

end

