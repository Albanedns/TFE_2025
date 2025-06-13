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
combinations = combinations[combinations['landmark']==1] # choisir PPI ou LPI (ici on garde LPI)

for _, row in combinations.iterrows():

    #on garde les données correspondantes 
    tree_count = row['tree_count']
    landmark = row['landmark']
    subset = df[(df['tree_count'] == tree_count) & (df['landmark'] == landmark)]
    
        
    subset= subset.dropna(subset=['rotation_error','aide_phare', 'AgeGroup', 'APOE', 'Gender', 'VideoGame', 'Education', 'Participant', 'VersionAppleGame'])
    print(len(subset['Participant'].unique()))

    #Standardisation
    scaler = StandardScaler()
    subset = subset[['rotation_error','aide_phare', 'AgeGroup', 'APOE', 'Gender', 'VideoGame', 'Education', 'Participant', 'VersionAppleGame']]
    subset[['VideoGame', 'Education','aide_phare']] = scaler.fit_transform(subset[['VideoGame', 'Education','aide_phare']])

    #Encodage des données catégorielles
    age_levels = ["<=50", "51-60", "61-70", ">70"]
    subset['Gender'] = subset['Gender'].astype('category')
    subset["AgeGroup"] = pd.Categorical(subset["AgeGroup"], categories=age_levels)
    subset['APOE'] = subset['APOE'].astype(str).astype('category')
        

    if subset.shape[0] > 1:
        
        try:
            df_r = pandas2ri.py2rpy(subset)
            ro.globalenv['df']= df_r
            ro.r('''
            library(lme4)
            # Ajuster le modèle mixte avec lme4
            library(lmerTest)
            mod_lmer <- lmer(rotation_error ~ APOE*aide_phare* AgeGroup + Gender + VideoGame + Education + (1|Participant) + (1|VersionAppleGame), data=df)
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
                     

                trend_estimates <- emtrends(mod_lmer, ~ 1, var = "aide_phare")
                trend_summary <- summary(trend_estimates)
                trend_df<- as.data.frame(trend_summary)
                 
                trend_df$t_value <- trend_df$aide_phare.trend / trend_df$SE
                trend_df$p_value <- 2 * pt(-abs(trend_df$t_value), df = trend_df$df)
                trend_df$adj.p.value <- p.adjust(trend_df$p_value, method = "holm")
                 
                trend_df$significance <- cut(trend_df$adj.p.value,
                             breaks = c(-Inf, 0.001, 0.01, 0.05, Inf),
                             labels = c("***", "**", "*", ""))
                print(trend_df)
                trend_estimates <- emtrends(mod_lmer, ~ APOE | AgeGroup, var = "aide_phare")
                trend_summary <- summary(trend_estimates)
                trend_df<- as.data.frame(trend_summary)
                 
                trend_df$t_value <- trend_df$aide_phare.trend / trend_df$SE
                trend_df$p_value <- 2 * pt(-abs(trend_df$t_value), df = trend_df$df)
                trend_df$adj.p.value <- p.adjust(trend_df$p_value, method = "holm")
                 
                trend_df$significance <- cut(trend_df$adj.p.value,
                             breaks = c(-Inf, 0.001, 0.01, 0.05, Inf),
                             labels = c("***", "**", "*", ""))
                print(trend_df)
                     
                # Comparer les pentes entre groupes d'âge
                trend_pairs <- emtrends(mod_lmer, pairwise ~ AgeGroup | APOE, var = "aide_phare")

                # Comparaisons par paire : est-ce que les effets de start_basket_distance diffèrent ?
                pairwise_trend_df <- as.data.frame(trend_pairs$contrasts)
                pairwise_trend_df$adj.p.value <- p.adjust(pairwise_trend_df$p.value, method = "holm")
                pairwise_trend_df$significance <- cut(pairwise_trend_df$adj.p.value,
                             breaks = c(-Inf, 0.001, 0.01, 0.05, Inf),
                             labels = c("***", "**", "*", ""))
                print(pairwise_trend_df)
                
                trend_pairs <- emtrends(mod_lmer, pairwise ~ APOE | AgeGroup , var = "aide_phare")

                # Comparaisons par paire : est-ce que les effets de start_basket_distance diffèrent ?
                pairwise_trend_df <- as.data.frame(trend_pairs$contrasts)
                pairwise_trend_df$adj.p.value <- p.adjust(pairwise_trend_df$p.value, method = "holm")
                pairwise_trend_df$significance <- cut(pairwise_trend_df$adj.p.value,
                             breaks = c(-Inf, 0.001, 0.01, 0.05, Inf),
                             labels = c("***", "**", "*", ""))
                print(pairwise_trend_df)
            ''')

                    
        except Exception as e:
            print(f"Model did not converge for tree_count={tree_count}, landmark={landmark}: {e}")
    else:
        print(f"Not enough data for tree_count={tree_count}, landmark={landmark}")