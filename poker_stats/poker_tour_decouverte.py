import os
from typing import List
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

path_data = os.getcwd()
diplomes = {'Master 2': 5, 'BTS': 2, 'BE': 0, 'Baccalauréat': 0, 'Bac+3': 3, 'BEP': -1, 'Bac+4': 4, 'Bac+2': 2,
            np.nan: np.nan, 'CAPES': 4, 'CAP': -1, 'Doctorat': 8, "Diplôme d'ingénieur": 5, 'DESS': 5,
            'CAPES (bac+4)': 4,
            'Master 2 Psychologie': 5, 'Bac': 0, 'BEp': -1, 'DUT': 2, 'DUG': 2, 'Bac+5': 5, 'Licence': 3, 'Bac pro': 0,
            'Bac+5 ingénieur': 5, 'BEPC': -2, 'Bac Pro': 0, 'DEA': 5, 'Bac+2 (DUT)': 2, 'Master': 5,
            'BEP comptable': -1,
            'Ingénieur': 5, 'Licence Mathématiques': -3, 'Master informatique': 5, 'Master biotechnologie': 5,
            'Aucun': -3, 'Diplôme Aide médico-psychologique': 2, 'BEP Eléctrotechnique': 2, 'Bac STG': 0, 'Master 1': 4,
            'Bac+5 Ingénieur': 5, 'DEUX': 2, 'CAP Maçonnerie': -1, 'Master Microbiologie': 5, 'Néant': -3, 'Bac STI': 0,
            'BE Educateur sportif': 0, 'Bac S': 0, 'Probatoire au DECS': 0, '0': 0, 'Licence informatique': 3,
            'BEP Commerce': 2, 'Maîtrise Droit des affaires': 4, 'Bac+2 Capacité Transport': 2, 'DUT SRC': 2,
            'BEP CAp': -1, 'MBA HEC': 5, 'CQP Bac+3': 3, 'sans': -3, 'BPAEA': "?", 'CQPM PSPA': "?", 'CFC': "?",
            'Bac+2 BTS MAI': 2, 'Bac D': 0, "Diplome d'Etat infirmier": 2, 'DECF': "?", 'Bac pro logistique': 0,
            'BP Génie Climatique': -1, 'BEP Agricole': -1, 'CAP vente': -1, 'Licence Pro': 3, 'Brevet': -2,
            'Licence info com journalisme': 3, 'Doctorat Chimie organique': 8, 'Licence pro': 3,
            'Master 2 Sciences de Gestion': 5, 'DEUG': 2, 'BEES': "?", 'Bac pro commerce': 0, 'Licence géographie': 3,
            'Bac pro Energie': 0, 'Mention complémentaire': "?", 'Maitrise': 4,
            'Master management et commerce international': 5, 'Diplome ingénieur': 5, 'Master 2 Droit': 5, 'PLP2': "?",
            'Licence Master': 5, 'Deug Staps': 2, 'Master 2 Gestion Environnement Industriel': 5, 'DCG': "?",
            'Bac pro secrétariat': 0, 'Bac+2 CPI': 2, 'BA Interior design': "?", 'Bac+5 Marketing': 5, 'DE': "?",
            'Permis moto': -3, 'BEP MSMA': -1, 'Ingénieur Bac+5': 5, 'BEP CAP': -2, 'Maîtrise': 4,
            'Maitrise sciences politique': 4, 'BTS Tourisme loisirs': 2, "Diplome d'ingénieur Bac+5": 5,
            'Maîtrise Bac+4': 4, "Diplôme d'Etat infirmière": 2, 'DUT GEA': 2, "Diplôme d'Etat": "?", 'CAp': -1,
            'Licence Anglais': 3, 'Licence Bac+3': 3, 'Bac sientifique': 0, 'DEES': "?", 'Brevet des collèges': -2,
            'Bac+2 BTS': 2, 'Bac BEES 1°': 1, 'Bac+2 / Coach certifié pro': 2, 'Niveau bac': 0, 'Niveaau Bac': 0,
            'CAFME': "?", 'Bac A': 0, 'Bac pro comptablité': 0, 'Diplôme Ingéniuer': 5, 'DEUST': 0,
            "Diplôme d'Etat Kinésithérapeute": 2, 'Bac s': 0, 'BTS MUC': 2, 'Brevet professionnel': -1, 'Bac ES': 0,
            'BPJEPS': "?", 'BEP ECASER': -1, 'Bac+5 Master': 5, 'Sans': -3, 'Deug': 2, 'Bac+11': 8, 'Master Bac+5': 5,
            'ITB': "?", 'Ostéopathe': "?", 'CAP plaquiste': -1, 'Bac C': 0, 'BAC': 0, 'BAC+4': 4, 'BTS commerce': 2,
            'Licence commerce': 3, 'Kinésithérapeute': 2, 'BPREA': "?", 'Bac+8 Doctorat': 8, 'Ingénieur du son': 5,
            'Licence Staps': 3, 'BTS informatique': 2, 'Bac+5 Diplôme ingénieur': 5, 'Master 2 Bac+5': 5,
            'DUT iniformatique': 2, 'CAP BEP': 2, 'Master finance': 5, 'BTS entreprise sncf': 2,
            'M1 école de commerce': 4, 'Licence Histoire': 3, 'BEP vente': 2, 'Bac pro élecronique': 0,
            "Diplôme d'Etat IADE (anesthésiste)": "?", 'Master de recherche': 5, 'BEP electrotechniques': 2,
            'Docteur en pharmacie': 8, 'Sans diplome': -3, 'CAP cuisinier': -1, 'DUT Gestion': 2, 'IUT': 2,
            'Bac CSP techn son': 0, 'Secondaire': -1, 'Autodidacte': -3, 'Bac pro électrotechnique': 0,
            'Master 2 Ostéopathe': 5, 'BTS Comptabilité Gestion': 2, 'DE infirmier': 2, 'Maitrise électricité': 4,
            'Cab+5': 5, 'Bep': 2, 'Bac Comptabilité': 0, 'DUT Mesures physiques': 2, "Diplôme d'infirmière": 2,
            'DE Assistante sociale': 2, 'BTS Industries agro-alimentaire': 2, 'BTS Professions Immobilières': 2,
            'Capacité': "?", 'Bac Général': 0, 'Licence Biologie': 3, 'DUT GMP': 2, 'M2': 5, 'DE kiné': 2,
            'DEUG Staps': 2, 'Bac+2 DUT': 2, 'Bac et BEES 1e degré': "?", 'BacPro': 0, 'BAJEPS': "?", 'Bac+1': 1,
            'Bac G2': 0, 'Maitrise Bac+4': 4, 'Bac+3 Licence Staps': 3, "Diplome d'ingénieur": 5}
CSP = {"Doctorante en science de l'éducation ; Animatrice": 'enseignant', 'Docker': 'ouvrier',
       'Educateur sportif': 'enseignant', 'Marin': 'fonctionnaire', 'Kinésithérapeute': 'médical',
       'Militaire': 'fonctionnaire',
       'Agent du service hospitalier': 'médical', "Responsable d'étude": '?', 'Infomaticien': 'informaticien',
       'Gérant': 'cadre', 'Manager': 'cadre', 'Artisan': 'artisan', 'Commercial': 'commercial',
       'Enseignant en physique-chimie': 'enseignant', 'Commercant': 'commercial', 'Etudiant': 'sans emploi',
       'Agent territoriale': 'fonctionnaire', 'Magasinier': 'commercial', 'Auto-entrepreneur': '?',
       np.nan: np.nan, 'Psychiatre': 'médical', 'Ingénieur développeur': 'informaticien', 'Serveuse': 'employé',
       'Employer de banque': 'employé', 'Enseignant': 'enseignant', 'Masseur-kinesithérapeute': 'médical',
       'Conseiller en orientation psychologue': 'fonctionnaire', 'Empoyée Patisserie': 'employé',
       'Gérant société maçonnerie': 'artisan', "Agent d'études": '?', 'Conseiller en gestion de patrimoine': 'employé',
       'Paysagiste': 'artisan', 'Informaticien': 'informaticien', 'Ingénieur informatique': 'informaticien',
       'Hôtesse de vente': 'commercial', 'Electricien': 'artisan', 'SAV Piscine': 'commercial', 'Horloger': 'artisan',
       'Homme au foyer': "sans emploi", 'Finance': 'cadre', 'Maçon': 'artisan', 'Auxiliaire médicale': 'médical',
       'Fonctionnaire': 'fonctionnaire', 'Paramédical': 'médical', 'Cuisinier': 'employé', 'Assistante': 'employé',
       'Technicien de laboratoire': 'employé', 'Statisticien': '?', 'Ascenseuriste': 'ouvrier',
       'Agent SNCF': 'fonctionnaire', 'Routier': 'ouvrier', 'Ingénieur monétique': 'cadre',
       'Employé municipal': 'fonctionnaire', 'Medecin': 'médical', "Chef d'équipe piste aéroportuaire": 'employé',
       'Employé de restauration': 'employé', 'Couvreur': 'artisan', 'Assistant de direction': 'employé',
       'Peintre en bâtiment': 'artisan', 'Superviseur péage': 'employé', 'Musicienne': 'artiste',
       'Chauffeur Poids lourd': 'ouvrier', 'Manager garage': 'cadre', "Ingénieur d'études": 'cadre',
       'Agent de maîtrise': 'cadre', 'Commerçant': 'commercial', 'Agent de Maîtrise': 'cadre',
       "Maître d'Education Physique": 'enseignant', 'VRp': "commercial", 'Commerce': "commercial",
       'Gérant société': "cadre", 'Détiticien': "médical", 'Conseiller Eclairagiste': "artisan",
       'Ressources Humaines': "employé", "Dirigeant d'entreprises": "cadre", 'Taxi': "employé",
       'Cadre commercial': "commercial", 'Médecin': "médical", 'Agent Territorial': "fonctionnaire",
       'Artisant': "artisan", "Chef d'entreprise": "cadre", 'Agent de consignation': "commercial",
       'Vendeuse': "commercial", 'Commerçant ambulant': "commercial", 'Aluminier': "artisan",
       'Responsable logistique': "employé", 'Moniteur de ski, commerçant': "enseignant",
       'Aide médico-psychologique': "médical", 'Agent de tourisme': "employé", 'Plombier': "artisan",
       'Menuisier': "artisan", 'Informatique': 'informaticien', "Ingénieur d'affaires": "cadre",
       'Technicien informatique': "informaticien", 'Intérimaire': "ouvrier", 'Directeur commercial': "cadre",
       'Gameplay programme': "employé", 'Infographiste': "employé", 'Patissier': "artisan",
       "Responsable d'auto-école": "cadre", 'Mécanicien auto': "ouvrier", 'Ingénieur informaticien': "informaticien",
       'Manutentionnaire': "ouvrier", 'Cadre': "cadre", 'Technicien de maintenance': "ouvrier", 'Livreur': "ouvrier",
       'Ingénieur': "cadre", 'Restaurateur': "artisan", 'Sans profession': "sans emploi",
       'Professuer': "enseignant",
       'Chef de quai': "employé", 'Educateur APS': "enseignant", 'Responsable logistique (agro alimentaire)': "employé",
       'Chef de chantier bâtiment': "artisan", "Ingénieur d'application (informatique)": "informaticien",
       "Agent d'exploitation spécialisé": "employé", 'Formateur': 'enseignant', 'Chef comptable': "cadre",
       'DAF': "cadre",
       'GRH': "employé", 'Expert monétique': "employé", 'Responsable Maintenance': "employé", 'Assureur': "employé",
       'Responsable Comptable': "employé", 'Evenementiel': "employé", 'Contrôleur': "?",
       'Chirurgien dentaire': "médical",
       'Mandataire de justice': "employé", 'Barman': "employé", 'Technicien vidéo': "employé",
       'Directeur': "cadre", 'Vendeur': "commercial", 'Serveur': "employé", 'Ingénieur bâtiment': "cadre",
       "Gérant d'entreprise": "cadre", 'Référent insertion pro': "fonctionnaire", 'naire': "fonctionnaire",
       'Logisticien': "employé", 'Chef de projet informatique': "informaticien", 'Responsable carroserie': "ouvrier",
       'Mécanicien aérostructure': "ouvrier", 'Assistant de gestion': "employé",
       'Fonctionnaire Trésor Public': "fonctionnaire",
       'Contrôleur territorial': "fonctionnaire", 'Grutier': "ouvrier", 'Directeur de projets': "cadre",
       'Consultant': "employé",
       'Gestionnaire de copropriété': "employé", 'Monitrice auto-école': "enseignant", 'Responsable rayon': "employé",
       'Technicien': "ouvrier", 'Ouvrier piscicole': "ouvrier", 'Cadre informatique': "informaticien",
       'Opérateur industriel': "ouvrier", 'Plombier chauffagiste': "artisan", 'Ouvrier polyvalent': "ouvrier",
       'Pizzaïolo': "employé", 'Indépendant': "?", 'Orthoprothésiste': "médical", 'Facteur': "fonctionnaire",
       'Pomppier': "fonctionnaire", 'Chauffeur Livreur': "ouvrier", 'Infirmier': "médical",
       'Assistant qualité': "employé",
       'AMQ': "?", 'Artisan plombier chauffagiste': "artisan", 'Agent commercial': "commercial", 'Sans': "sans emploi",
       'Magasinier cariste': "ouvrier", 'Postier': "fonctionnaire", 'Ouvrier': "ouvrier",
       'préparateur de commandes': "employé", 'Technicien HSE': "employé", 'Chargée de projet': "employé",
       'Opérateur production plasturgie': "ouvrier", 'Doctorat Chimie organique': "enseignant",
       'Chauffeur livreur': "ouvrier", 'Géomètre': "employé", 'Agent administratif': "fonctionnaire",
       'Chimiste': "employé",
       "Technicien d'Atelier": "ouvrier", 'Cheminot': "fonctionnaire", 'Conseiller Gestion de patrimoine': "employé",
       'Plombier électricien': "artisan", 'Chauffeur mécanicien poids lourd': "ouvrier", 'Technicien labo': "employé",
       'Cariste': "ouvrier", 'Aide soignant': "médical", 'VRP': "commercial", 'Dispatelier': "ouvrier",
       "Joueur de billard d'équipe de France": "sportif", 'Gestionnaire achat': "employé", 'Chef de projet': "employé",
       'Acheteur': "commercial", 'Entrepreneur': "cadre", 'Manager rayon': "employé",
       'Professeur physique chimie': "enseignant",
       'Employé Grande distribution': "employé", "Chef d'équipe": "employé", 'Artisan Peintre décorateur': "artisan",
       'Brancardier': "médical", 'Téléopératrice': "employé", 'Directeur technique': "cadre",
       'Aide médico psychologique': "médical", 'Technicien SAV': "employé", 'Stagiaire': "sans emploi",
       'Chauffeur': "ouvrier", 'Employé de laboratoire': "employé", 'Infirmière': "médical",
       'Chirurgien dentiste': "médical", 'Préparateur de commande': "employé",
       "Chargé d'Affaire Ingenierie": "commercial",
       'Joueur de poker': "joueur poker", 'Directeur de projet': "employé", 'Commerical': "commercial",
       'A. ED': "enseignant", 'Instituteur': "enseignant", 'sans': "sans emploi", 'Web designer': "employé",
       'Technicien piscine': "ouvrier", 'Graphiste': "employé", 'Coopération internationale': "employé",
       'Comptable': "employé", "Assistant d'éducation": "enseignant",
       'Fonctionnaire Surveillant Pénitentiaire': "fonctionnaire",
       'Conducteur du Trésor': "fonctionnaire", 'Agent de sécurité': "employé", 'Assistante de direction': "employé",
       'Boulanger': "artisan", "Ingénieur d'étude aéronautique": "cadre", 'Conducteur de machine': "ouvrier",
       'Acteur': "artiste", 'Infirmière Libérale': "médical", 'Conseiller financier': "employé",
       'Informaticienne': "informaticien", '0': "sans emploi", 'Auditeur': "employé", 'Employé Dermo': "employé",
       'Gestionnaire Ressources humaines': "employé", 'Assistant gestion': "employé", 'cadre commercial': "commercial",
       'Technicien logistique': "employé", 'Contrôleur arérien': "employé", 'Educateur': "enseignant",
       'Opérateur de sureté aéroportuaire': "employé", 'Chef de cuisine': "employé", 'Ambulancier': "médical",
       "Chargé d'activités": "employé", 'Dessinateur en batiment': "employé",
       "Technicien supérieur à l'agence de l'eau": "fonctionnaire", 'Hydraulicien': "employé",
       'Technicien aéronautique': "employé", 'Agent de voyage': "commercial", "Ingéniuer d'étude": "cadre",
       'Moniteur de golf': "enseignant", 'Technicien de maintenance aéronautique': "employé",
       'Responsable commerciale': "commercial", 'Responsable qualité': "employé",
       'Agent Administratif': "fonctionnaire",
       'Ingénieur en informatique': "informaticien", 'Convoyeur de fond': "ouvrier", 'Professeur': "enseignant",
       'Habillage avion': "ouvrier", 'Recherche emploi': "sans emploi", 'RM': "employé", 'Expert immobilier': "employé",
       'Ingéniuer informatique': "informaticien", 'Sapeur pompier': "fonctionnaire", 'Esthéticienne': "employé",
       'Conducteur offset': "ouvrier", 'Equipier polyvalent Gérant une équipe': "employé",
       'Chargé de communication': "employé", 'Restauration': "employé", 'Etudiant L3 droit': "sans emploi",
       'Chef de rayon': "employé", "Chargé d'affaires": "employé", 'Directeur financier': "cadre", 'Artiste': "artiste",
       'CAIC': "ouvrier", 'Educateur spécialisé': "enseignant", 'Agent de tri': "employé",
       'Adjoint administratif': "fonctionnaire", 'Exploitant auto école': "enseignant", 'Imprimeur': "artisan",
       'Carriste': "ouvrier", 'Préparateur': "employé", 'Responsable Assemblage Satellite': "employé",
       'Mécanicien moto': "artisan", 'Monteur-contrôleur': "ouvrier", 'Vendeur technique': "commercial",
       'agent immobilier': "commercial", 'Aide soignante': "médical", 'Fonctionnaire police': "fonctionnaire",
       'Vendeur SNCF': "commercial", "Bâtiment bureau d'études": "employé", "Chargé d'études": "employé",
       'Conseiller clientèle': "commercial", 'Radiologue': "médical", 'artisan patissier': "artisan",
       'développeur': "informaticien", 'Peintre': "artisan", 'Boucher': "artisan",
       "Agent d'opération voiture": "ouvrier",
       'Ingénieur aérospatial': "cadre", 'Cadre administratif': "cadre", 'Kinésithérapeute-ostéopathe': "médical",
       'CG': "?", 'Musicien': "artiste", "Adjoint d'animation": "employé", 'Pompe funèbre': "artisan",
       "Chef d'atelier adjoint": "ouvrier", 'Agent immobilier': "commercial", 'Opticien': "médical",
       'Vendeur GSM': "commercial", "Chef d'équipe Maçon": "artisan", 'Agent de prévention incendie': "employé",
       "Ingénieur d'étude": "cadre", 'Responsable R&D dans SS2': "informaticien", 'Cadre de santé': "médical",
       'Employé du secteur privé': "employé", 'Menuisier aluminuim PUC': "artisan", 'Renovation de maison': "artisan",
       'Employé libre service': "employé", "Technicien bureau d'étude aéronautique": "employé", 'Sécurité': "employé",
       'Pharmacien': "médical", 'Libéral': "?", 'Maraicher': "agriculteur", 'MNS': "?", 'Bac+2': "?",
       "Professeur d'EPS": "enseignant", 'Ingéniuer': "cadre", 'Technicien gestion de production': "employé",
       'Photographe': "artiste", 'Gendarme': "fonctionnaire", 'Administratuer Réseau': "informaticien", 'Salarié': "?",
       'Technicien livreur': "ouvrier", 'Responsable Technique': "cadre", 'Animation': "employé",
       'Chef de ligne usine': "ouvrier", 'Ascensoriste': "ouvrier", 'Conseiller commercial': "commercial",
       'Agent immbolier': "commercial", 'Infirmier libéral': "médical", 'Ingénieur commercial': "commercial",
       'Ingénieur électronique': "cadre", 'Responsable magasin': "cadre", 'Employé': "employé",
       'Gérant Constructeur de maisons individuelle': "cadre", 'Psychomotricien': "médical",
       'Technicien Eléctronique': "ouvrier", 'Contrôleur de gestion': "employé", 'Travail Sncf': "fonctionnaire",
       'Agent de conduite SNCF': "fonctionnaire", "Juriste d'entreprise": "employé", 'Employé dans un golf': "employé",
       'Stagière attachée de presse': "sans emploi", 'Etudiante Master en Histoire patrimoine': "sans emploi",
       'Outilleur automobile': "ouvrier", 'Moniteur auto-école': "enseignant", 'Tatoueuse': "artiste",
       'Agent logistique': "employé", 'Agent de maitrise': "employé", 'Responsable agence immobilière': "commercial",
       'INF anesthésiste': "médical", 'Ingénieur SAV': "cadre", 'Doctorant en biologie': "enseignant",
       'Responsable expédition': "employé", 'Attaché commerciale': "commercial", 'Commerciale': "commercial",
       'Electro-mécanicien': "ouvrier", "Hôtesse d'accueil": "employé", 'Principal de collège': "enseignant",
       'DRH': "cadre",
       'Radioprotection': "médical", 'Développeur informatique': "informaticien", '3': "?", 'Evènementiel': "employé",
       'Moniteur de plongée': 'enseignant', 'Auto entrepreneur': "?", 'Conseiller': "employé",
       "Assistante d'éducation": "enseignant", 'Gardiennage': "employé", 'Agriculteur': "agriculteur",
       "Animateur / directeur d'acceuil de loisirs": "employé", 'Technico commercial': "commercial",
       'Fabricant cosmétique': "employé", 'Baby sitter': "employé", 'Directeur général': "cadre",
       'Sous-officier : informaticien': "informaticien", 'Chaudronnier': "ouvrier", 'Agent de voyages': "employé",
       'Fonctionnaire de police': "fonctionnaire", 'Electronicien': "ouvrier", "Directeur d'agence": "cadre",
       'Ingénieur en électronique informatique': "informaticien", 'Agent technique': "employé",
       'Téléconseiller': "employé",
       'Technicien clientème ERDF': "employé", 'Apprenti': "ouvrier", 'Frigoriste': "ouvrier", 'Ostéopathe': "médical",
       'Ingénieur système': "cadre", 'Dirigeant de société': "cadre", 'Ouvrier en papeterie': "ouvrier",
       'Fonction publique': "fonctionnaire", 'Technicien électronique': "ouvrier", 'INfirmier': "médical",
       'Electricien Mécanicien': "ouvrier", 'CDI': "?", 'formaticien': "informaticien", 'commercial': "commercial",
       'Ingénieur méthodes et développement': "cadre", 'Directeur associé': "cadre", 'Ingénieur télécom': "cadre",
       'Directeur de restaurant': "cadre", 'Animateur socail': "employé", 'Agent admiinistratif': "fonctionnaire",
       'Contrôleur de travaux': "employé", 'Technicien chimiste': "employé", 'Conducteur de travaux': "employé",
       'Technicien Développement': "employé", 'Assistante sociale': "fonctionnaire", 'Chef Comptable': "employé",
       'Agent de pasteurisation': "ouvrier", 'Opticienne': "médical", 'Sapeur pompier de Paris': "fonctionnaire",
       'Hotelerie': "employé", 'Calorifugeur': "ouvrier", 'Consultant en investissement': "employé",
       'Cadre territorial': "fonctionnaire", 'Militzire': "fonctionnaire", 'Chargé de clientèle': "commercial",
       'Ouvrier qualifié': "ouvrier", 'Dirigeant': "cadre", 'Technicien de Maintenance industrielle': "ouvrier",
       'Exploitant transport': "employé", 'Professeur des écoles': 'enseignant', "agent d'exploitation": 'ouvrier',
       'Technicien qualité': "employé", 'Ingénieur, informatique': "informaticien",
       'Masseur kinésithérapeute': "médical",
       'Cinéaste': "artiste", 'Hrologer': "artisan", 'Exploitant agricol': "agriculteur",
       'Coordonnateurr Sécurité et Protection de la Santé': "employé", 'Soudeur': "ouvrier",
       'Controleur de gestion': "employé", 'Gestionnaire informatique': "informaticien",
       'Enseignant de conduite': 'enseignant', 'Vendeur magasinier': "commercial", 'Agent de la poste': "fonctionnaire",
       'Technicien sup': "employé", 'Agent de tri, cariste': "ouvrier",
       'Chargé de placement et développement commercial': "commercial", 'Moniteur de tennis': "enseignant",
       "Directeur d'agence bancaire": "cadre", 'Fonctionnaire (gardien de la paix)': "fonctionnaire",
       'Gérant de société': "cadre", 'Vendeur voiture': "commercial", 'Demonstrateur pour Lacoste': "employé",
       'Secrétaire': "employé", 'Technicien packaging': "ouvrier", 'Responsable technique': "employé",
       'Chef de projet en informatique': "informaticien", 'Employé de banque': "employé", 'Maintenance': "ouvrier",
       'Consultant informatique freelance': "informaticien", "Agent d'accueil": "employé", 'Responsable DAO': "employé",
       'Cadre Fonction Publique': "fonctionnaire", 'Consultant en informatique': "informaticien",
       "Agent d'exploitation": "ouvrier", 'Journaliste': "employé", 'Entraineur de football': "sportif",
       'Charge de formation': "enseignant", 'Responsable commercial': "commercial",
       "Ministère de l'intérieur": "fonctionnaire", 'Agent de manoeuvre sncf': "fonctionnaire", 'Animateur': "employé",
       'Vernisseur': "ouvrier", 'Directeur administratif': "cadre", 'assistant Gestion': "employé",
       'Responsable secteur careolier': "ouvrier"}
sommes = {'Moins de 5E': 2.5, 'Entre 10 et 20E': 15, 'Entre 5 et 10E': 7.5, 'Aucune': 0, 'Entre 20 et 50E': 35,
          'Plus de 50E': 75, '0': 0}


def import_and_code_fpt() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(path_data, "FPT_all.csv"), sep="\t", encoding="utf-8")
    cols = [col for col in df.columns if df.loc[~df.loc[:, col].isna(), :].shape[0] > 100
            and "code" not in col and "Unnamed" not in col]
    df = df.loc[:, cols]
    df.loc[:, "sites"] = df.loc[:, "sites"].apply(lambda z: str(z).replace(", ", " ").replace(" ; ", " ")
                                                  .replace(",", " "))
    df.loc[:, "nb_sites"] = df.loc[:, "sites"].apply(lambda z: len(z.split()))
    df.loc[df.loc[:, "sites"] == "3-4 sites", "nb_sites"] = 4
    df.loc[df.loc[:, "sites"] == "Presque tous", "nb_sites"] = 10
    df.loc[df.loc[:, "sites"].isin(["Tous", "TOus", "tous"]), "nb_sites"] = 10
    df.loc[:, "annees_etudes"] = df.loc[:, "diplome"].apply(lambda z: diplomes[z])
    df.loc[:, "csp"] = df.loc[:, "profession"].apply(lambda z: CSP[z])
    df.loc[:, "1e_somme_misee"] = df.loc[:, "1somme"].apply(lambda z: sommes[z])
    df.loc[:, "somme_compte_cont"] = df.loc[:, "somme_compte"].apply(lambda z: float(str(z).replace(",", ".")))
    df.loc[:, "age"] = df.loc[:, "naissance"].apply(lambda z: 2011-int(z))

    df.to_csv(os.path.join(path_data, "temp_fact.csv"), encoding="utf-8", sep=";")
    for col in df.columns:
        print(col)
        print(df.loc[:, col].unique())
    print(df.groupby("nb_sites").nb_sites.count())
    print(df.groupby("annees_etudes").annees_etudes.count())
    print(df.groupby("csp").csp.count())

    return df


def vs_tab(df_all: pd.DataFrame, field_y: str, fields_x: List[str]) -> None:
    df = df_all.loc[~df_all.loc[:, field_y].isna()]
    modalites_y = sorted(list(df.loc[:, field_y].unique()))
    for field in fields_x:
        modalites_x = [x for x in list(df.loc[:, field].unique()) if df.loc[df.loc[:, field] == x, :].shape[0] >= 10]
        tab = pd.DataFrame(columns=modalites_y + [field])
        for n_e in modalites_x:
            x = [df.loc[(df.loc[:, field_y] == n_s) & (df.loc[:, field] == n_e)].shape[0] for n_s in modalites_y] + [
                n_e]
            tab = tab.append(pd.Series(x, index=modalites_y + [field]), ignore_index=True)

        tab = tab.loc[:, [field] + modalites_y].set_index(field)
        print(tab)


def vs_plot(df_all: pd.DataFrame, field_y: str, fields_x: List[str]) -> None:
    df = df_all.loc[~df_all.loc[:, field_y].isna()]
    for field in fields_x:
        modalites_x = [x for x in list(df.loc[:, field].unique()) if df.loc[df.loc[:, field] == x, :].shape[0] >= 20]
        plt.figure()
        plt.plot(modalites_x, [df.loc[df.loc[:, field] == a, field_y].mean() for a in modalites_x])
        plt.xlabel(f"{field}")
        plt.title(f"Distribution {field_y}")
        plt.show()


if __name__ == "__main__":
    df_fpt = import_and_code_fpt()
    vs_tab(df_fpt, "somme_compte", ["annees_etudes", "csp", "age"])
    vs_plot(df_fpt, "somme_compte_cont", ["annees_etudes", "csp", "age"])
