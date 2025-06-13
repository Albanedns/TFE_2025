import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Activer la conversion automatique pandas <-> R
pandas2ri.activate()

# Charger les packages R
emmeans = importr("emmeans")
base = importr("base")
ggplot2 = importr("ggplot2")
lme4 = importr("lme4")


df = pd.read_csv('errors.csv')

combinations = df[['tree_count', 'landmark']].drop_duplicates()
combinations = combinations[combinations['tree_count']==1] #Nombre d'arbre dans les essais pris en compte 
combinations = combinations[combinations['landmark']==0] # choisir PPI ou LPI 
for _, row in combinations.iterrows():

     #on garde les données correspondantes 
    tree_count = row['tree_count']
    landmark = row['landmark']
    subset = df[(df['tree_count'] == tree_count) & (df['landmark'] == landmark)]
    
        
    subset= subset.dropna(subset=['path_length_ratio_1', 'AgeGroup', 'APOE', 'Gender', 'VideoGame', 'Education', 'Participant', 'VersionAppleGame'])
    
    #Standardisation
    scaler = StandardScaler()
    subset = subset[['path_length_ratio_1', 'AgeGroup', 'APOE', 'Gender', 'VideoGame', 'Education', 'Participant', 'VersionAppleGame']]
    subset[['VideoGame', 'Education']] = scaler.fit_transform(subset[['VideoGame', 'Education']])
        
    #Encodage des données catégorielles
    age_levels = ["<=50", "51-60", "61-70", ">70"]
    subset['Gender'] = subset['Gender'].astype('category')
    subset["AgeGroup"] = pd.Categorical(subset["AgeGroup"], categories=age_levels)
    subset['APOE'] = subset['APOE'].astype(str).astype('category')
    #subset["VideoGame"] = np.log1p(subset["VideoGame"] - subset["VideoGame"].min() + 1)
        
        

    if subset.shape[0] > 1:
        
        try:
            df_r = pandas2ri.py2rpy(subset)
            ro.globalenv['df']= df_r
            ro.r('''
            library(lme4)
            # Ajuster le modèle mixte avec lme4
            library(lmerTest)
                
            mod_lmer <- lmer(path_length_ratio_1 ~ AgeGroup * APOE + Gender + VideoGame + Education + (1|Participant) + (1|VersionAppleGame), data=df)
            print(summary(mod_lmer))
            ''')
            # Utiliser emmeans pour obtenir les effets marginaux
            ro.r('''
            library(emmeans)
            library(ggplot2)
            library(ggpubr)
            library(ggsignif)

            contrast_results <- contrast(emmeans(mod_lmer, ~ AgeGroup), method = "pairwise")
            contrast_apoe <- contrast(emmeans(mod_lmer, ~ APOE| AgeGroup), method = "pairwise")
            contrast_df <- as.data.frame(contrast_results)
            contrast_df_apoe <- as.data.frame(contrast_apoe)
            contrast_df$adj.p.value <- p.adjust(contrast_df$p.value, method = "holm")
            contrast_df_apoe$adj.p.value <- p.adjust(contrast_df_apoe$p.value, method = "holm")
            print(contrast_df_apoe)
                     
            get_stars <- function(p) {
                if (p < 0.001) return("***")
                else if (p < 0.01) return("**")
                else if (p < 0.05) return("*")
                else return("")
            }
                
            contrast_df$sts <- vapply(contrast_df$adj.p.value, get_stars, character(1))
            contrast_df$sts <- as.character(contrast_df$sts)
            print(contrast_df)
            contrast_df <- contrast_df[contrast_df$sts != "", ]
            group_x <- c("<=50" = 1, "51-60" = 2, "61-70" = 3, ">70" = 4)
            print(group_x)
            clean_age <- function(s) {
                gsub("[()]", "", s)
            }
                     
            str(contrast_df$sts)
            annot_age <- contrast_df %>%
                mutate(
                    age1 = clean_age(sub(" - .*", "", contrast)),  # avant le tiret
                    age2 = clean_age(sub(".* - ", "", contrast)),  # après le tiret
                    x = group_x[age1],
                    xend = group_x[age2],
                    y = 2000+ seq_len(nrow(contrast_df)) * 200,  # hauteur à ajuster si besoin
                    mid_x = (x + xend)/2
                ) 
            str(annot_age$sts)
            print(annot_age)

            # Créez le graphique emmip
            plot <- emmip(mod_lmer, APOE ~ AgeGroup, CIs = TRUE, type = "response",
                                linearg = list(linetype = "dashed"), CIarg = list(lwd = 3, alpha = 0.4), 
                                xlab = "Age (years)", ylab = " Path Length Ratio ") +
                ggtitle("PPI: Path Length Ratio ") +
                theme_bw() +
                    scale_color_manual(values = c("gray65", "gray10"), labels = c("0.0" = expression(epsilon[4] ~"non carriers"),
             "1.0" = expression(epsilon[4] ~ "carriers")), name = "APOE") +
                theme(panel.ontop = FALSE,
                        axis.text.x = element_text(size = 16,face = "plain"),
                        axis.text.y = element_text(size = 16,face = "plain"),
                        axis.title = element_text(size = 20, face = "plain"),
                        legend.text = element_text(size = 16),
                        legend.title = element_text(size = 16),
                        plot.title = element_text(size = 18, face = 'bold'),
                        legend.position = "right") +
                coord_cartesian(ylim = c(0, 2.5), clip = "off")
                         
            plot <- plot + geom_segment(data = annot_age,
                aes(x = x, xend = xend, y = y, yend = y),
                linewidth = 0.8, inherit.aes = FALSE) +
                geom_segment(data = annot_age,
            aes(x = x, xend = x, y = y+40, yend = y - 40),
            linewidth = 0.8,
            inherit.aes = FALSE)+
                geom_segment(data = annot_age,
               aes(x = xend, xend = xend, y = y+40, yend = y - 40),
               linewidth = 0.8,
               inherit.aes = FALSE)+ 
                geom_text(data = annot_age, aes(x = mid_x, y = y+10 , label = sts),
                    size = 9, show.legend = FALSE,inherit.aes = FALSE)
            plot <- plot + 
                annotate("segment", x = 2.9, xend = 3.1, y = 2, yend = 2,linewidth = 0.8) +  # segment horizontal
                annotate("segment", x = 2.9, xend = 2.9, y = 2+0.08, yend = 2-0.08,linewidth = 0.8) +
                annotate("segment", x = 3.1, xend = 3.1, y = 2+0.08, yend = 2-0.08,linewidth = 0.8) +
                annotate("text", x = 3, y = 2.08, label = "*", size = 9)

            ggsave(filename = "Plot/PLR1_PPI.png", plot = plot, width = 8, height = 6)
            ''')


        except Exception as e:
            print(f"Model did not converge for tree_count={tree_count}, landmark={landmark}: {e}")
    else:
        print(f"Not enough data for tree_count={tree_count}, landmark={landmark}")