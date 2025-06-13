import statsmodels.formula.api as smf
import seaborn as sns
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt
from pymer4.models import Lmer
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

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
combinations = combinations[combinations['landmark']==1] # choisir PPI ou LPI

for _, row in combinations.iterrows():

     #on garde les données correspondantes 
    tree_count = row['tree_count']
    landmark = row['landmark']
    subset = df[(df['tree_count'] == tree_count) & (df['landmark'] == landmark)]

    subset = subset.dropna(subset=['response_distance','incoming_distance', 'start_basket_distance', 'AgeGroup', 'APOE', 'Gender', 'VideoGame', 'Education', 'Participant', 'VersionAppleGame'])
    subset = subset[['response_distance','incoming_distance', 'start_basket_distance', 'AgeGroup', 'APOE', 'Gender', 'VideoGame', 'Education', 'Participant', 'VersionAppleGame']]
    
    #Standardisation
    scaler= StandardScaler()
    subset[['VideoGame', 'Education']] = scaler.fit_transform(subset[['VideoGame', 'Education']])

    #Encodage des données catégorielles
    age_levels = ["<=50", "51-60", "61-70", ">70"]
    subset['Gender'] = subset['Gender'].astype('category')
    subset["AgeGroup"] = pd.Categorical(subset["AgeGroup"], categories=age_levels, ordered=True)
    subset['APOE'] = subset['APOE'].astype(str).astype('category')
    

    if subset.shape[0] > 1:
        try:
            
            df_r = pandas2ri.py2rpy(subset)
            ro.globalenv['df']= df_r
            ro.r('''
                library(lme4)
                library(lmerTest)
                # Ajuster le modèle mixte avec lme4
                mod_lmer <- lmer(response_distance ~  Gender + VideoGame + Education +  start_basket_distance*AgeGroup + incoming_distance*AgeGroup+ incoming_distance*AgeGroup*APOE + start_basket_distance*AgeGroup*APOE+ (1|Participant) + (1|VersionAppleGame), data=df)
                print(summary(mod_lmer))
            ''')
                # Utiliser emmeans pour obtenir les effets marginaux
            ro.r('''
                library(emmeans)
                library(ggplot2)
                library(ggpubr)
                library(ggsignif)
                library(tidyr)
                library(dplyr)
                trend_incoming <- emtrends(mod_lmer, ~ AgeGroup * APOE, var = "incoming_distance", method = "pairwise")
                result_pairs <- pairs(trend_incoming, by = "AgeGroup")
                result_pairs <- as.data.frame(summary(result_pairs))
                result_pairs$adj.p.value <- p.adjust(result_pairs$p.value, method = "holm")
                print(result_pairs)
                
                # Obtenir les pentes de start_basket_distance par groupe d'âge
                age_contrast <- contrast(emmeans(mod_lmer, ~ AgeGroup), method = "pairwise")
                age_contrast <- as.data.frame(age_contrast)
                age_contrast$adj.p.value <- p.adjust(age_contrast$p.value, method = "holm")
                print(age_contrast)
                trend_estimates <- emtrends(mod_lmer, ~ AgeGroup, var = "incoming_distance")
                trend_summary <- summary(trend_estimates)
                trend_df<- as.data.frame(trend_summary)
                 
                trend_df$t_value <- trend_df$incoming_distance.trend / trend_df$SE
                trend_df$p_value <- 2 * pt(-abs(trend_df$t_value), df = trend_df$df)
                trend_df$adj.p.value <- p.adjust(trend_df$p_value, method = "holm")
                 
                trend_df$significance <- cut(trend_df$adj.p.value,
                             breaks = c(-Inf, 0.001, 0.01, 0.05, Inf),
                             labels = c("***", "**", "*", ""))
                trend_df <- trend_df[trend_df$significance != "", ]
                trend_df$group1 <- trend_df$AgeGroup
                print(trend_df)
                emmip_data <- emmip(mod_lmer, AgeGroup ~ incoming_distance,
                    at = list(incoming_distance = 10500),
                    CIs = TRUE, type = "response", plotit = FALSE)
                print(emmip_data)
                star_annotations  <- trend_df %>%
                    left_join(emmip_data %>% select(group1 = AgeGroup, y1 = yvar), by = "group1") %>%
                    mutate(
                        x = 8500,
                        y = c(6200, 4900, 3535, 3100),
                        label = significance
                )
                
                 
                # Comparer les pentes entre groupes d'âge
                trend_pairs <- emtrends(mod_lmer, pairwise ~ AgeGroup, var = "incoming_distance")

                # Comparaisons par paire : est-ce que les effets de start_basket_distance diffèrent ?
                pairwise_trend_df <- as.data.frame(trend_pairs$contrasts)
                pairwise_trend_df$adj.p.value <- p.adjust(pairwise_trend_df$p.value, method = "holm")
                pairwise_trend_df$significance <- cut(pairwise_trend_df$adj.p.value,
                             breaks = c(-Inf, 0.001, 0.01, 0.05, Inf),
                             labels = c("***", "**", "*", ""))
                print(pairwise_trend_df)
                pairwise_trend_df <- pairwise_trend_df[pairwise_trend_df$significance != "", ]
                print(pairwise_trend_df)
                
                emmip_data <- emmip(mod_lmer, AgeGroup ~ incoming_distance,
                    at = list(incoming_distance = 8000),
                    CIs = TRUE, type = "response", plotit = FALSE)
            
                # Créez le graphique emmip
                plot <- emmip(mod_lmer, AgeGroup ~ incoming_distance, 
                 at = list(incoming_distance = seq(0, 8000, by = 420)), CIs = TRUE, type = "response",
                                linearg = list(linetype = "solid", size = 2), CIarg = list(lwd = 4, alpha = 0.4), 
                                xlab = "Incoming distance (v m)", ylab = "Response distance (v m)") +
                    ggtitle("") +
                    theme_bw() +
                    theme(axis.text.x = element_text(size = 14),
                            axis.text.y = element_text(size = 14),
                             axis.text = element_text(size = 16),
                            axis.title = element_text(size = 16),
                            legend.text = element_text(size = 16),
                            legend.title = element_text(size = 16),
                            title = element_text(size = 18, face = 'bold'),
                            plot.margin = margin(t = 10, r = 150, b = 10, l = 10),
                            legend.position = "left") +
                         coord_cartesian(xlim = c(0, 8000), ylim = c(0, 8000), clip = "off")
                 plot <- plot + 
                    geom_text(data = star_annotations, aes(x = x, y = y, label = label, color = AgeGroup),
                    size = 7, show.legend = FALSE,inherit.aes = FALSE)
                
                clean_age <- function(s) {
                    gsub("[()]", "", s)
                }

# Séparer les groupes comparés
                bars <- pairwise_trend_df %>%
                        mutate(group1 = clean_age(sub(" - .*", "", contrast)),  # avant le tiret
                        group2 = clean_age(sub(".* - ", "", contrast)),
                        label = significance)

                # Fusionner avec les valeurs de fin de courbe (à start_basket_distance = 8000)
                bars <- bars %>%
                    left_join(emmip_data %>% select(group1 = AgeGroup, y1 = yvar), by = "group1") %>%
                    left_join(emmip_data %>% select(group2 = AgeGroup, y2 = yvar), by = "group2") %>%
                    mutate(
                        x = 8700 + seq_len(nrow(bars)) * 500,
                        y_start = pmin(y1, y2),
                        y_end = pmax(y1, y2),
                        label_y = (y_start + y_end)/2# marge au-dessus pour l'astérisque
                )
                print(bars)
                plot <- plot +
                    geom_segment(data = bars, aes(x = x, xend = x, y = y_start, yend = y_end), linewidth = 0.9, inherit.aes = FALSE) +
                    geom_segment(data = bars, aes(x = x+100, xend = x-100, y = y_start, yend = y_start), linewidth = 0.9, inherit.aes = FALSE) +
                 geom_segment(data = bars, aes(x = x+100, xend = x-100, y = y_end, yend = y_end), linewidth = 0.9, inherit.aes = FALSE) +
                    geom_text(data = bars, aes(x = x + 300, y = label_y, label = label), size = 6,angle = 90,inherit.aes = FALSE)
                ggsave(filename = "Plot/LPI_incoming_distance.png", plot = plot, width = 8, height = 6)
            ''')

        
        except Exception as e:
            print(f"Model did not converge for tree_count={tree_count}, landmark={landmark}: {e}")
    else:
        print(f"Not enough data for tree_count={tree_count}, landmark={landmark}")