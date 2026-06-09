# ==============================================================================
# PROJECT: EndoDecay-Sim v13.1 (Computational Core Engine)
# AUTHOR: Muhammet Yagiz Zavrak
# STATUS: Independent Research / Open Science Initiative
# PARADIGMS: Milstein SDE Scheme, Active Feedback Topology, True Paillier Crypto
# DEPENDENCIES: numpy, pandas, scikit-learn, phe
# COPYRIGHT: (c) 2026 Muhammet Yagiz Zavrak. All rights reserved.
# ==============================================================================

import json
import numpy as np
import pandas as pd
from phe import paillier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class FHIRLocalParser:
    """Lokal EHR kayıtlarındaki LOINC kodlu FHIR JSON verilerini ayrıştırır."""
    @staticmethod
    def parse_bundle(fhir_json_string):
        try:
            bundle = json.loads(fhir_json_string)
            extracted = {}
            for entry in bundle.get('entry', []):
                res = entry.get('resource', {})
                if res.get('resourceType') == 'Observation':
                    code = res.get('code', {}).get('coding', [{}])[0].get('code')
                    val = res.get('valueQuantity', {}).get('value')
                    if code == '2571-8':   extracted['HbA1c'] = val
                    elif code == '62238-1': extracted['Baseline_eGFR'] = val
                    elif code == '33762-6': extracted['Baseline_NT_proBNP'] = val
            return {
                'Age': bundle.get('patient_age', 55.0),
                'HbA1c': extracted.get('HbA1c', 6.5),
                'Baseline_eGFR': extracted.get('Baseline_eGFR', 90.0),
                'Baseline_NT_proBNP': extracted.get('Baseline_NT_proBNP', 100.0),
                'Toxic_Insult': bundle.get('environmental_exposure', 0.2),
                'scRNA_Pseudotime': bundle.get('transcriptomic_damage_index', 0.3),
                'Longitudinal_Updates': bundle.get('longitudinal_records', [])
            }
        except Exception as e:
            print(f"CRITICAL: FHIR parsing failed. Error: {str(e)}")
            return None


class HomomorphicEncryptionProtocol:
    """FedAvg döngüleri için gerçek toplamsal homomorfik Paillier kriptosistemi."""
    @staticmethod
    def generate_keypair():
        """Klinik merkezler ve ana sunucu için 2048-bit asimetrik anahtar üretir."""
        return paillier.generate_paillier_keypair(n_length=2048)

    @staticmethod
    def encrypt_weights(public_key, weight_list):
        """Model ağırlık vektörünü public key ile gerçek anlamda şifreler (Ciphertext)."""
        return [public_key.encrypt(float(w)) for w in weight_list]

    @staticmethod
    def aggregate_ciphertexts(encrypted_vectors):
        """Şifreli verileri deşifre etmeden merkezi sunucuda toplar."""
        base_vector = list(encrypted_vectors[0])
        for next_vector in encrypted_vectors[1:]:
            for i in range(len(base_vector)):
                base_vector[i] = base_vector[i] + next_vector[i]
        return base_vector


class PhysiologicalFeedbackTopology:
    """Organ sistemleri arasındaki anlık diferansiyel geri besleme ağ yapısı."""
    def __init__(self):
        self.nodes = {
            'Heart': {'NT_proBNP_load': 100.0},
            'Kidney': {'eGFR_clearance': 90.0},
            'Liver': {'Metabolic_detox': 1.0}
        }
        self.cardiorenal_axis_weight = 0.85
        self.hepatorenal_axis_weight = 0.40

    def compute_timestep_cross_talk(self, instantaneous_damage_vector, dt):
        mean_damage = np.mean(instantaneous_damage_vector)
        self.nodes['Heart']['NT_proBNP_load'] += mean_damage * self.cardiorenal_axis_weight * dt
        self.nodes['Kidney']['eGFR_clearance'] -= mean_damage * self.cardiorenal_axis_weight * dt
        self.nodes['Kidney']['eGFR_clearance'] = max(self.nodes['Kidney']['eGFR_clearance'], 5.0)
        
        kidney_stress = max(0.0, (90.0 - self.nodes['Kidney']['eGFR_clearance']) * 0.01)
        self.nodes['Liver']['Metabolic_detox'] += kidney_stress * self.hepatorenal_axis_weight * dt
        return self.nodes['Kidney']['eGFR_clearance']


class EndoDecaySimV13_1Core:
    def __init__(self, cohort_size=10000, seed=42):
        self.cohort_size = cohort_size
        self.seed = seed
        np.random.seed(self.seed)
        self.feedback_network = PhysiologicalFeedbackTopology()
        
    def generate_synthetic_cohort(self):
        means = [55, 6.8, 0, 0, 0.4, 0.5]
        cov = [
            [100, 2.1, -0.3, 0.4, 2.0, 0.4],  
            [2.1, 1.4, -0.4, 0.5, 0.3, 0.20], 
            [-0.3, -0.4, 1.0, -0.4, -0.1, -0.3],  
            [0.4, 0.5, -0.4, 1.0, 0.2, 0.4],  
            [2.0, 0.3, -0.1, 0.2, 0.25, 0.05],   
            [0.4, 0.20, -0.3, 0.4, 0.05, 0.04]   
        ]
        cov_matrix = 0.5 * (np.array(cov) + np.array(cov).T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        cov_fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        raw_data = np.random.multivariate_normal(means, cov_fixed, self.cohort_size)
        
        self.cohort = pd.DataFrame({
            'Age': np.clip(raw_data[:, 0], 18.0, 105.0),
            'HbA1c': np.clip(raw_data[:, 1], 3.5, 18.0),
            'Baseline_eGFR': np.clip(95.0 - np.exp(raw_data[:, 2] * 1.1) * 12.0, 5.0, 180.0),
            'Baseline_NT_proBNP': np.clip(80.0 + np.exp(raw_data[:, 3] * 1.4) * 65.0, 5.0, 35000.0),
            'Toxic_Insult': np.clip(raw_data[:, 4], 0.0, 1.0),
            'scRNA_Pseudotime': np.clip(raw_data[:, 5], 0.0, 1.0)
        })

    def run_milstein_sde_engine(self, total_months=120, base_dt=0.05, noise_intensity=0.015):
        current_time = 0.0
        egfr = self.cohort['Baseline_eGFR'].values.copy()
        nt_probnp = self.cohort['Baseline_NT_proBNP'].values.copy()
        E = np.ones(self.cohort_size) * 100.0
        
        event_observed = np.zeros(self.cohort_size, dtype=int)
        time_to_event = np.ones(self.cohort_size) * total_months
        
        while current_time < total_months:
            drift_gradient = -0.002 * (self.cohort['scRNA_Pseudotime'].values * 1.5) * E
            max_velocity = np.max(np.abs(drift_gradient))
            dt = np.minimum(base_dt, 0.01 / (max_velocity + 1e-5))
            dt = np.maximum(dt, 0.001)
            
            live_kidney = self.feedback_network.compute_timestep_cross_talk(100.0 - E, dt)
            
            egfr_decay = np.where(E < 85.0, 1.0 - (0.0004 * (100.0 - E) * dt * (90.0 / live_kidney)), 1.0)
            egfr = np.clip(egfr * egfr_decay, 5.0, 180.0)
            nt_surge = np.where(egfr < 60.0, 1.0 + (0.0008 * (60.0 - egfr) * dt), 1.0)
            nt_probnp = np.clip(nt_probnp * nt_surge, 5.0, 35000.0)
            
            collapsed = (egfr < 30.0) | (nt_probnp > 1000.0)
            first_collapse = collapsed & (event_observed == 0)
            event_observed[first_collapse] = 1
            time_to_event[first_collapse] = current_time
            
            dW = np.random.normal(0, np.sqrt(dt), size=self.cohort_size)
            milstein_correction = 0.5 * (noise_intensity ** 2) * E * (dW ** 2 - dt)
            E += (drift_gradient * dt) + (noise_intensity * E * dW) + milstein_correction
            E = np.clip(E, 0.0, 100.0)
            current_time += dt
            
        self.cohort['Event_Observed'] = event_observed
        self.cohort['Time_To_Event'] = time_to_event
        self.cohort['Endothelial_Retention_Final'] = E
        self.cohort['eGFR_Final'] = egfr
        self.cohort['NT_proBNP_Final'] = nt_probnp

    def train_isolated_prognostic_classifier(self):
        features = ['Age', 'HbA1c', 'Baseline_eGFR', 'Baseline_NT_proBNP', 'Toxic_Insult', 'scRNA_Pseudotime']
        X = self.cohort[features].values
        y = self.cohort['Event_Observed'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        clf.fit(X_train, y_train)
        return clf

    def export_survival_curves(self):
        timeline = np.arange(0, 121, 1)
        survival_probabilities = []
        active_at_risk = self.cohort_size
        for month in timeline:
            events = self.cohort[(self.cohort['Event_Observed'] == 1) & (self.cohort['Time_To_Event'].astype(int) == month)].shape[0]
            if active_at_risk > 0:
                step = 1.0 - (events / active_at_risk)
                if len(survival_probabilities) == 0: survival_probabilities.append(step)
                else: survival_probabilities.append(survival_probabilities[-1] * step)
            else:
                survival_probabilities.append(0.0)
            active_at_risk -= events
        return pd.DataFrame({'Timeline_Month': timeline, 'Survival_Probability': survival_probabilities})


if __name__ == "__main__":
    print("STATUS: Initializing mechanistic solver and population matrix generation...")
    core = EndoDecaySimV13_1Core(cohort_size=10000)
    core.generate_synthetic_cohort()
    core.run_milstein_sde_engine()
    core.train_isolated_prognostic_classifier()
    
    core.cohort.to_csv('EndoDecay_Sim_v13_Final_Data.csv', index=False)
    curves = core.export_survival_curves()
    curves.to_csv('EndoDecay_Sim_v13_Survival_Data.csv', index=False)
    print("STATUS: Physical CSV records successfully deployed to system storage.")
    
    print("\n================ HOMOMORPHIC CRYPTO VALIDATION WORKFLOW ================")
    pub_key, priv_key = HomomorphicEncryptionProtocol.generate_keypair()
    
    node_harvard_weights = [0.3440, 0.2034, 0.1985]
    node_oxford_weights  = [0.3410, 0.2050, 0.1970]
    
    cipher_harvard = HomomorphicEncryptionProtocol.encrypt_weights(pub_key, node_harvard_weights)
    cipher_oxford  = HomomorphicEncryptionProtocol.encrypt_weights(pub_key, node_oxford_weights)
    
    global_aggregated_cipher = HomomorphicEncryptionProtocol.aggregate_ciphertexts([cipher_harvard, cipher_oxford])
    
    decrypted_global_sum = [priv_key.decrypt(x) for x in global_aggregated_cipher]
    print(f"Decrypted Global FedAvg Vector Sum: {decrypted_global_sum}")
    print("========================================================================")