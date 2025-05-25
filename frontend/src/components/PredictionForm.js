import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import axios from 'axios';
import { FiUser, FiClock, FiActivity, FiHeart, FiTrendingUp, FiMoon, FiShield } from 'react-icons/fi';

const FormContainer = styled.div`
  max-width: 800px;
  margin: 2rem auto;
  padding: 2rem;
`;

const FormCard = styled(motion.div)`
  background: white;
  border-radius: 20px;
  padding: 3rem;
  box-shadow: ${props => props.theme.shadows.large};
  border: 1px solid ${props => props.theme.colors.border};
`;

const Title = styled.h1`
  text-align: center;
  color: ${props => props.theme.colors.primary};
  margin-bottom: 2rem;
  font-size: 2.5rem;
  font-weight: 700;
`;

const FormGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  margin-bottom: 2rem;
`;

const FormGroup = styled.div`
  display: flex;
  flex-direction: column;
`;

const Label = styled.label`
  font-weight: 600;
  color: ${props => props.theme.colors.text};
  margin-bottom: 0.5rem;
  font-size: 1rem;
`;

const Input = styled.input`
  padding: 1rem;
  border: 2px solid ${props => props.theme.colors.border};
  border-radius: 10px;
  font-size: 1rem;
  transition: all 0.3s ease;
  background: white;

  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.primary};
    box-shadow: 0 0 0 3px ${props => props.theme.colors.primary}20;
  }

  &:invalid {
    border-color: #e74c3c;
  }

  &:disabled {
    background: #f8f9fa;
    color: #6c757d;
  }
`;

const Select = styled.select`
  padding: 1rem;
  border: 2px solid ${props => props.theme.colors.border};
  border-radius: 10px;
  font-size: 1rem;
  background: white;
  transition: all 0.3s ease;

  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.primary};
    box-shadow: 0 0 0 3px ${props => props.theme.colors.primary}20;
  }
`;

const CheckboxContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.5rem;
`;

const Checkbox = styled.input`
  width: 1.2rem;
  height: 1.2rem;
  accent-color: ${props => props.theme.colors.primary};
`;

const CheckboxLabel = styled.label`
  font-size: 0.9rem;
  color: ${props => props.theme.colors.textSecondary};
  cursor: pointer;
`;

const SubmitButton = styled(motion.button)`
  width: 100%;
  background: linear-gradient(135deg, ${props => props.theme.colors.primary}, ${props => props.theme.colors.secondary});
  color: white;
  padding: 1.2rem 2rem;
  border: none;
  border-radius: 50px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  margin-top: 2rem;
  box-shadow: ${props => props.theme.shadows.medium};
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-2px);
    box-shadow: ${props => props.theme.shadows.large};
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

const LoadingSpinner = styled.div`
  display: inline-block;
  width: 20px;
  height: 20px;
  border: 3px solid #ffffff;
  border-radius: 50%;
  border-top-color: transparent;
  animation: spin 1s ease-in-out infinite;
  margin-right: 10px;

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
`;

const ErrorMessage = styled.div`
  background: #fee;
  border: 1px solid #fcc;
  color: #c33;
  padding: 1rem;
  border-radius: 10px;
  margin-top: 1rem;
  text-align: center;
`;

const HelpText = styled.small`
  color: ${props => props.theme.colors.textSecondary};
  margin-top: 0.25rem;
  font-size: 0.85rem;
`;

const InfoBox = styled.div`
  background: #e3f2fd;
  border: 1px solid #bbdefb;
  border-radius: 8px;
  padding: 1rem;
  margin-top: 0.5rem;
  font-size: 0.85rem;
  color: #1565c0;
`;

const PredictionForm = () => {
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [bpUnknown, setBpUnknown] = useState(false);

    const [formData, setFormData] = useState({
        gender: '',
        age: '',
        occupation: '',
        sleep_duration: '',
        quality_of_sleep: '',
        physical_activity_level: '',
        stress_level: '',
        bmi_category: '',
        blood_pressure_systolic: '',
        blood_pressure_diastolic: '',
        heart_rate: '',
        daily_steps: ''
    });

    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
        setError('');
    };

    const handleBpUnknownChange = (e) => {
        setBpUnknown(e.target.checked);
        if (e.target.checked) {
            setFormData({
                ...formData,
                blood_pressure_systolic: 'N/A',
                blood_pressure_diastolic: 'N/A'
            });
        } else {
            setFormData({
                ...formData,
                blood_pressure_systolic: '',
                blood_pressure_diastolic: ''
            });
        }
        setError('');
    };

    const validateForm = () => {
        const requiredFields = Object.keys(formData);
        for (let field of requiredFields) {
            if (!formData[field]) {
                setError(`Please fill in all fields. Missing: ${field.replace('_', ' ')}`);
                return false;
            }
        }

        // Validate ranges (skip blood pressure if N/A)
        if (formData.age < 18 || formData.age > 100) {
            setError('Age must be between 18 and 100');
            return false;
        }
        if (formData.sleep_duration < 3 || formData.sleep_duration > 12) {
            setError('Sleep duration must be between 3 and 12 hours');
            return false;
        }
        if (formData.quality_of_sleep < 1 || formData.quality_of_sleep > 10) {
            setError('Quality of sleep must be between 1 and 10');
            return false;
        }

        // Validate blood pressure if not N/A
        if (!bpUnknown) {
            if (formData.blood_pressure_systolic < 80 || formData.blood_pressure_systolic > 200) {
                setError('Systolic blood pressure must be between 80 and 200 mmHg');
                return false;
            }
            if (formData.blood_pressure_diastolic < 40 || formData.blood_pressure_diastolic > 120) {
                setError('Diastolic blood pressure must be between 40 and 120 mmHg');
                return false;
            }
        }

        return true;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!validateForm()) return;

        setLoading(true);
        setError('');

        try {
            // Use production API URL - replace with your actual Render URL
            const API_URL = process.env.REACT_APP_API_URL || 'https://sleepview-app.onrender.com';

            const response = await axios.post(`${API_URL}/predict`, formData, {
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            navigate('/results', { state: { prediction: response.data, formData } });
        } catch (err) {
            setError(err.response?.data?.error || 'An error occurred while making the prediction. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const occupations = [
        'Software Engineer', 'Doctor', 'Sales Representative', 'Teacher', 'Nurse',
        'Engineer', 'Accountant', 'Scientist', 'Lawyer', 'Salesperson', 'Manager'
    ];

    return (
        <FormContainer>
            <FormCard
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
            >
                <Title>Sleep Health Assessment</Title>

                <form onSubmit={handleSubmit}>
                    <FormGrid>
                        <FormGroup>
                            <Label htmlFor="gender">Gender</Label>
                            <Select
                                id="gender"
                                name="gender"
                                value={formData.gender}
                                onChange={handleChange}
                                required
                            >
                                <option value="">Select Gender</option>
                                <option value="Male">Male</option>
                                <option value="Female">Female</option>
                            </Select>
                        </FormGroup>

                        <FormGroup>
                            <Label htmlFor="age">Age</Label>
                            <Input
                                type="number"
                                id="age"
                                name="age"
                                value={formData.age}
                                onChange={handleChange}
                                min="18"
                                max="100"
                                required
                            />
                            <HelpText>Age in years (18-100)</HelpText>
                        </FormGroup>

                        <FormGroup>
                            <Label htmlFor="occupation">Occupation</Label>
                            <Select
                                id="occupation"
                                name="occupation"
                                value={formData.occupation}
                                onChange={handleChange}
                                required
                            >
                                <option value="">Select Occupation</option>
                                {occupations.map(occ => (
                                    <option key={occ} value={occ}>{occ}</option>
                                ))}
                            </Select>
                        </FormGroup>

                        <FormGroup>
                            <Label htmlFor="sleep_duration">Sleep Duration (hours)</Label>
                            <Input
                                type="number"
                                id="sleep_duration"
                                name="sleep_duration"
                                value={formData.sleep_duration}
                                onChange={handleChange}
                                min="3"
                                max="12"
                                step="0.1"
                                required
                            />
                            <HelpText>Average hours of sleep per night (3-12)</HelpText>
                        </FormGroup>

                        <FormGroup>
                            <Label htmlFor="quality_of_sleep">Quality of Sleep (1-10)</Label>
                            <Input
                                type="number"
                                id="quality_of_sleep"
                                name="quality_of_sleep"
                                value={formData.quality_of_sleep}
                                onChange={handleChange}
                                min="1"
                                max="10"
                                required
                            />
                            <HelpText>Rate your sleep quality (1=Poor, 10=Excellent)</HelpText>
                        </FormGroup>

                        <FormGroup>
                            <Label htmlFor="physical_activity_level">Physical Activity Level (1-100)</Label>
                            <Input
                                type="number"
                                id="physical_activity_level"
                                name="physical_activity_level"
                                value={formData.physical_activity_level}
                                onChange={handleChange}
                                min="1"
                                max="100"
                                required
                            />
                            <HelpText>Weekly physical activity level</HelpText>
                        </FormGroup>

                        <FormGroup>
                            <Label htmlFor="stress_level">Stress Level (1-10)</Label>
                            <Input
                                type="number"
                                id="stress_level"
                                name="stress_level"
                                value={formData.stress_level}
                                onChange={handleChange}
                                min="1"
                                max="10"
                                required
                            />
                            <HelpText>Rate your stress level (1=Low, 10=High)</HelpText>
                        </FormGroup>

                        <FormGroup>
                            <Label htmlFor="bmi_category">BMI Category</Label>
                            <Select
                                id="bmi_category"
                                name="bmi_category"
                                value={formData.bmi_category}
                                onChange={handleChange}
                                required
                            >
                                <option value="">Select BMI Category</option>
                                <option value="Underweight">Underweight</option>
                                <option value="Normal">Normal</option>
                                <option value="Overweight">Overweight</option>
                                <option value="Obese">Obese</option>
                            </Select>
                        </FormGroup>

                        <FormGroup>
                            <Label htmlFor="blood_pressure_systolic">Systolic Blood Pressure</Label>
                            <Input
                                type="number"
                                id="blood_pressure_systolic"
                                name="blood_pressure_systolic"
                                value={bpUnknown ? 'N/A' : formData.blood_pressure_systolic}
                                onChange={handleChange}
                                min="80"
                                max="200"
                                disabled={bpUnknown}
                                required
                            />
                            <CheckboxContainer>
                                <Checkbox
                                    type="checkbox"
                                    id="bp_unknown"
                                    checked={bpUnknown}
                                    onChange={handleBpUnknownChange}
                                />
                                <CheckboxLabel htmlFor="bp_unknown">
                                    I don't know my blood pressure
                                </CheckboxLabel>
                            </CheckboxContainer>
                            {!bpUnknown && <HelpText>Systolic BP (mmHg) - the top number</HelpText>}
                            {bpUnknown && (
                                <InfoBox>
                                    We'll use age-appropriate average values for your blood pressure in the analysis.
                                </InfoBox>
                            )}
                        </FormGroup>

                        <FormGroup>
                            <Label htmlFor="blood_pressure_diastolic">Diastolic Blood Pressure</Label>
                            <Input
                                type="number"
                                id="blood_pressure_diastolic"
                                name="blood_pressure_diastolic"
                                value={bpUnknown ? 'N/A' : formData.blood_pressure_diastolic}
                                onChange={handleChange}
                                min="40"
                                max="120"
                                disabled={bpUnknown}
                                required
                            />
                            {!bpUnknown && <HelpText>Diastolic BP (mmHg) - the bottom number</HelpText>}
                        </FormGroup>

                        <FormGroup>
                            <Label htmlFor="heart_rate">Heart Rate (BPM)</Label>
                            <Input
                                type="number"
                                id="heart_rate"
                                name="heart_rate"
                                value={formData.heart_rate}
                                onChange={handleChange}
                                min="40"
                                max="120"
                                required
                            />
                            <HelpText>Resting heart rate (beats per minute)</HelpText>
                        </FormGroup>

                        <FormGroup>
                            <Label htmlFor="daily_steps">Daily Steps</Label>
                            <Input
                                type="number"
                                id="daily_steps"
                                name="daily_steps"
                                value={formData.daily_steps}
                                onChange={handleChange}
                                min="0"
                                max="30000"
                                required
                            />
                            <HelpText>Average daily steps</HelpText>
                        </FormGroup>
                    </FormGrid>

                    {error && <ErrorMessage>{error}</ErrorMessage>}

                    <SubmitButton
                        type="submit"
                        disabled={loading}
                        whileHover={{ scale: loading ? 1 : 1.02 }}
                        whileTap={{ scale: loading ? 1 : 0.98 }}
                    >
                        {loading && <LoadingSpinner />}
                        {loading ? 'Analyzing...' : 'Get Prediction'}
                    </SubmitButton>
                </form>
            </FormCard>
        </FormContainer>
    );
};

export default PredictionForm; 
