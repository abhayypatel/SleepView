import React from 'react';
import { useLocation, Link } from 'react-router-dom';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import {
  FiCheckCircle, FiAlertTriangle, FiAlertCircle, FiArrowLeft,
  FiHeart, FiMoon, FiActivity, FiClock, FiShield, FiTrendingUp,
  FiUser, FiBarChart3, FiInfo
} from 'react-icons/fi';

const ResultsContainer = styled.div`
  max-width: 1000px;
  margin: 2rem auto;
  padding: 2rem;
`;

const ResultCard = styled(motion.div)`
  background: white;
  border-radius: 20px;
  padding: 3rem;
  box-shadow: ${props => props.theme.shadows.large};
  border: 1px solid ${props => props.theme.colors.border};
  margin-bottom: 2rem;
`;

const Title = styled.h1`
  text-align: center;
  color: ${props => props.theme.colors.primary};
  margin-bottom: 2rem;
  font-size: 2.5rem;
  font-weight: 700;
`;

const PredictionSection = styled.div`
  text-align: center;
  margin-bottom: 3rem;
  padding: 2rem;
  background: ${props => props.riskLevel === 'None' ? '#d4edda' : props.riskLevel === 'Sleep Apnea' ? '#fff3cd' : '#f8d7da'};
  border-radius: 15px;
  border: 2px solid ${props => props.riskLevel === 'None' ? '#c3e6cb' : props.riskLevel === 'Sleep Apnea' ? '#ffeaa7' : '#f5c6cb'};
`;

const RiskLevel = styled.h2`
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 1rem;
  color: ${props => props.riskLevel === 'None' ? '#155724' : props.riskLevel === 'Sleep Apnea' ? '#856404' : '#721c24'};
`;

const RiskIcon = styled.div`
  font-size: 4rem;
  margin-bottom: 1rem;
`;

const Confidence = styled.p`
  font-size: 1.2rem;
  color: ${props => props.theme.colors.textSecondary};
  margin-bottom: 1rem;
`;

const Description = styled.p`
  font-size: 1.1rem;
  line-height: 1.6;
  color: ${props => props.theme.colors.text};
`;

const DetailsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  margin-bottom: 3rem;
`;

const DetailCard = styled.div`
  background: #f8f9fa;
  padding: 1.5rem;
  border-radius: 15px;
  border: 1px solid ${props => props.theme.colors.border};
`;

const DetailTitle = styled.h3`
  color: ${props => props.theme.colors.primary};
  margin-bottom: 1rem;
  font-size: 1.3rem;
`;

const DetailItem = styled.div`
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.5rem;
  padding: 0.5rem 0;
  border-bottom: 1px solid #e9ecef;

  &:last-child {
    border-bottom: none;
  }
`;

const DetailLabel = styled.span`
  font-weight: 500;
  color: ${props => props.theme.colors.text};
`;

const DetailValue = styled.span`
  color: ${props => props.theme.colors.textSecondary};
`;

const RecommendationsSection = styled.div`
  margin-bottom: 3rem;
`;

const RecommendationCard = styled.div`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 2rem;
  border-radius: 15px;
  margin-bottom: 2rem;
`;

const RecommendationTitle = styled.h3`
  font-size: 1.5rem;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const RecommendationList = styled.ul`
  list-style: none;
  padding: 0;
`;

const RecommendationItem = styled.li`
  margin-bottom: 0.75rem;
  padding-left: 1.5rem;
  position: relative;

  &:before {
    content: '✓';
    position: absolute;
    left: 0;
    color: #4ade80;
    font-weight: bold;
  }
`;

const ActionButtons = styled.div`
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
`;

const ActionButton = styled(motion(Link))`
  display: inline-block;
  padding: 1rem 2rem;
  border-radius: 50px;
  text-decoration: none;
  font-weight: 600;
  font-size: 1rem;
  transition: all 0.3s ease;
  text-align: center;
  min-width: 150px;
`;

const PrimaryButton = styled(ActionButton)`
  background: linear-gradient(135deg, ${props => props.theme.colors.primary}, ${props => props.theme.colors.secondary});
  color: white;
  box-shadow: ${props => props.theme.shadows.medium};

  &:hover {
    transform: translateY(-2px);
    box-shadow: ${props => props.theme.shadows.large};
  }
`;

const Results = () => {
  const location = useLocation();
  const { prediction, formData } = location.state || {};

  if (!prediction) {
    return (
      <ResultsContainer>
        <ResultCard>
          <Title>No Results Found</Title>
          <p style={{ textAlign: 'center' }}>
            Please complete the assessment first.
          </p>
          <ActionButtons>
            <PrimaryButton to="/predict">Take Assessment</PrimaryButton>
          </ActionButtons>
        </ResultCard>
      </ResultsContainer>
    );
  }

  const getRiskInfo = (disorder) => {
    switch (disorder) {
      case 'None':
        return {
          icon: '😴',
          color: '#155724',
          description: 'Great news! Based on your responses, you appear to have healthy sleep patterns with no significant indicators of sleep disorders.'
        };
      case 'Sleep Apnea':
        return {
          icon: '😰',
          color: '#856404',
          description: 'Your responses suggest potential indicators of sleep apnea, a condition where breathing repeatedly stops and starts during sleep.'
        };
      case 'Insomnia':
        return {
          icon: '😵',
          color: '#721c24',
          description: 'Your responses indicate potential signs of insomnia, characterized by difficulty falling asleep, staying asleep, or both.'
        };
      default:
        return {
          icon: '❓',
          color: '#6c757d',
          description: 'Unable to determine sleep disorder risk from the provided information.'
        };
    }
  };

  const getRecommendations = (disorder) => {
    const baseRecommendations = [
      'Maintain a consistent sleep schedule',
      'Create a relaxing bedtime routine',
      'Keep your bedroom cool, dark, and quiet',
      'Limit screen time before bed',
      'Avoid caffeine and large meals before bedtime'
    ];

    switch (disorder) {
      case 'Sleep Apnea':
        return [
          'Consult a sleep specialist for proper diagnosis',
          'Consider a sleep study if recommended',
          'Maintain a healthy weight',
          'Sleep on your side instead of your back',
          'Avoid alcohol and sedatives before bed',
          ...baseRecommendations
        ];
      case 'Insomnia':
        return [
          'Practice relaxation techniques like meditation',
          'Try cognitive behavioral therapy for insomnia (CBT-I)',
          'Limit daytime naps to 20-30 minutes',
          'Get regular exercise, but not close to bedtime',
          'Consider keeping a sleep diary',
          ...baseRecommendations
        ];
      default:
        return [
          'Continue your healthy sleep habits',
          'Monitor any changes in sleep patterns',
          'Stay physically active during the day',
          ...baseRecommendations
        ];
    }
  };

  const riskInfo = getRiskInfo(prediction.prediction);
  const recommendations = getRecommendations(prediction.prediction);

  return (
    <ResultsContainer>
      <ResultCard
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <Title>Your Sleep Health Results</Title>

        <PredictionSection riskLevel={prediction.prediction}>
          <RiskIcon>{riskInfo.icon}</RiskIcon>
          <RiskLevel riskLevel={prediction.prediction}>
            {prediction.prediction === 'None' ? 'Healthy Sleep Pattern' : prediction.prediction}
          </RiskLevel>
          <Confidence>
            Confidence: {(prediction.confidence * 100).toFixed(1)}%
          </Confidence>
          <Description>{riskInfo.description}</Description>
        </PredictionSection>

        <DetailsGrid>
          <DetailCard>
            <DetailTitle>📊 Your Input Summary</DetailTitle>
            <DetailItem>
              <DetailLabel>Age:</DetailLabel>
              <DetailValue>{formData.age} years</DetailValue>
            </DetailItem>
            <DetailItem>
              <DetailLabel>Gender:</DetailLabel>
              <DetailValue>{formData.gender}</DetailValue>
            </DetailItem>
            <DetailItem>
              <DetailLabel>Sleep Duration:</DetailLabel>
              <DetailValue>{formData.sleep_duration} hours</DetailValue>
            </DetailItem>
            <DetailItem>
              <DetailLabel>Sleep Quality:</DetailLabel>
              <DetailValue>{formData.quality_of_sleep}/10</DetailValue>
            </DetailItem>
            <DetailItem>
              <DetailLabel>Stress Level:</DetailLabel>
              <DetailValue>{formData.stress_level}/10</DetailValue>
            </DetailItem>
            <DetailItem>
              <DetailLabel>BMI Category:</DetailLabel>
              <DetailValue>{formData.bmi_category}</DetailValue>
            </DetailItem>
          </DetailCard>

          <DetailCard>
            <DetailTitle>🎯 Risk Probabilities</DetailTitle>
            {prediction.probabilities && Object.entries(prediction.probabilities).map(([disorder, prob]) => (
              <DetailItem key={disorder}>
                <DetailLabel>{disorder === 'None' ? 'Healthy' : disorder}:</DetailLabel>
                <DetailValue>{(prob * 100).toFixed(1)}%</DetailValue>
              </DetailItem>
            ))}
          </DetailCard>
        </DetailsGrid>

        <RecommendationsSection>
          <RecommendationCard>
            <RecommendationTitle>
              💡 Personalized Recommendations
            </RecommendationTitle>
            <RecommendationList>
              {recommendations.map((rec, index) => (
                <RecommendationItem key={index}>{rec}</RecommendationItem>
              ))}
            </RecommendationList>
          </RecommendationCard>
        </RecommendationsSection>

        <ActionButtons>
          <PrimaryButton
            to="/predict"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Take Another Assessment
          </PrimaryButton>
        </ActionButtons>
      </ResultCard>
    </ResultsContainer>
  );
};

export default Results; 