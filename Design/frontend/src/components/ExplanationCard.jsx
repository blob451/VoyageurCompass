import React, { useState, useRef, useEffect } from 'react';
import { llmLogger, explanationLogger } from '../utils/logger';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  IconButton,
  Chip,
  Collapse,
  CircularProgress,
  Alert,
  Divider,
  Tooltip,
  Select,
  MenuItem,
  FormControl,
  InputLabel
} from '@mui/material';
import {
  Psychology,
  ExpandMore,
  ExpandLess,
  Refresh,
  SmartToy,
  CheckCircle,
  Error,
  Warning,
  Info
} from '@mui/icons-material';

const ExplanationCard = ({
  title,
  analysisId,
  explanation = null,
  isLoading = false,
  error = null,
  onGenerate,
  onRefresh,
  defaultExpanded = false,
  showControls = true,
  variant = 'standard', // 'summary', 'standard', 'detailed'
  confidence = null,
  method = null,
  timestamp = null,
  children
}) => {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [selectedDetail, setSelectedDetail] = useState(variant);
  const requestTimeoutRef = useRef(null);
  const lastRequestRef = useRef(null);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (requestTimeoutRef.current) {
        clearTimeout(requestTimeoutRef.current);
      }
    };
  }, []);

  const handleExpand = () => {
    setExpanded(!expanded);
  };

  const handleGenerate = () => {
    if (onGenerate && !isLoading) {
      // Prevent multiple rapid requests
      const now = Date.now();
      if (lastRequestRef.current && (now - lastRequestRef.current) < 1000) {
        llmLogger.debounce('Generate request', 'too soon after previous request');
        return;
      }
      
      // Clear any pending requests
      if (requestTimeoutRef.current) {
        clearTimeout(requestTimeoutRef.current);
      }
      
      // Log the request initiation
      explanationLogger.workflow(analysisId, 'Request initiated', { 
        detailLevel: selectedDetail, 
        type: 'generate' 
      });
      
      // Debounce the request
      requestTimeoutRef.current = setTimeout(() => {
        lastRequestRef.current = Date.now();
        llmLogger.request(analysisId, selectedDetail, 'generate');
        onGenerate(selectedDetail);
      }, 100);
    }
  };

  const handleRefresh = () => {
    if (onRefresh && !isLoading) {
      // Prevent multiple rapid requests
      const now = Date.now();
      if (lastRequestRef.current && (now - lastRequestRef.current) < 1000) {
        llmLogger.debounce('Refresh request', 'too soon after previous request');
        return;
      }
      
      // Clear any pending requests
      if (requestTimeoutRef.current) {
        clearTimeout(requestTimeoutRef.current);
      }
      
      // Log the refresh initiation
      explanationLogger.workflow(analysisId, 'Refresh initiated', { 
        detailLevel: selectedDetail, 
        type: 'refresh' 
      });
      
      // Debounce the request
      requestTimeoutRef.current = setTimeout(() => {
        lastRequestRef.current = Date.now();
        llmLogger.request(analysisId, selectedDetail, 'refresh');
        onRefresh(selectedDetail);
      }, 100);
    }
  };

  const getConfidenceColor = (conf) => {
    if (!conf) return 'default';
    if (conf >= 0.8) return 'success';
    if (conf >= 0.6) return 'warning';
    return 'error';
  };

  const getConfidenceIcon = (conf) => {
    if (!conf) return <Info />;
    if (conf >= 0.8) return <CheckCircle />;
    if (conf >= 0.6) return <Warning />;
    return <Error />;
  };

  const getMethodIcon = (method) => {
    if (method === 'llama') return <SmartToy />;
    return <Psychology />;
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '';
    try {
      const date = new Date(timestamp);
      return date.toLocaleString();
    } catch (error) {
      return '';
    }
  };

  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Psychology color="primary" />
            <Typography variant="h6">
              {title}
            </Typography>
            {method && (
              <Tooltip title={method === 'llama' ? 'LLaMA 3.1 70B Generated' : 'Template Generated'}>
                <Chip 
                  icon={getMethodIcon(method)}
                  label={method === 'llama' ? 'AI' : 'Template'}
                  size="small"
                  color={method === 'llama' ? 'primary' : 'default'}
                  variant="outlined"
                />
              </Tooltip>
            )}
            {confidence !== null && (
              <Tooltip title={`Confidence: ${(confidence * 100).toFixed(1)}%`}>
                <Chip
                  icon={getConfidenceIcon(confidence)}
                  label={`${(confidence * 100).toFixed(0)}%`}
                  size="small"
                  color={getConfidenceColor(confidence)}
                  variant="outlined"
                />
              </Tooltip>
            )}
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {showControls && (
              <>
                <FormControl size="small" sx={{ minWidth: 100 }}>
                  <InputLabel>Detail</InputLabel>
                  <Select
                    value={selectedDetail}
                    label="Detail"
                    onChange={(e) => setSelectedDetail(e.target.value)}
                    disabled={isLoading}
                  >
                    <MenuItem value="summary">Summary</MenuItem>
                    <MenuItem value="standard">Standard</MenuItem>
                    <MenuItem value="detailed">Detailed</MenuItem>
                  </Select>
                </FormControl>
                
                {!explanation && (
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={handleGenerate}
                    disabled={isLoading || !analysisId}
                    startIcon={isLoading ? <CircularProgress size={16} /> : <SmartToy />}
                  >
                    Generate
                  </Button>
                )}
                
                {explanation && (
                  <IconButton
                    onClick={handleRefresh}
                    disabled={isLoading}
                    color="primary"
                    size="small"
                    title="Regenerate Explanation"
                  >
                    {isLoading ? <CircularProgress size={20} /> : <Refresh />}
                  </IconButton>
                )}
              </>
            )}
            
            <IconButton
              onClick={handleExpand}
              size="small"
              color="primary"
            >
              {expanded ? <ExpandLess /> : <ExpandMore />}
            </IconButton>
          </Box>
        </Box>

        {/* Content */}
        <Collapse in={expanded}>
          {error && (() => {
            // Log error details for debugging
            explanationLogger.error('Explanation generation failed', {
              analysisId,
              errorStatus: error.status,
              errorMessage: error.data?.error || error.message,
              errorType: error.constructor.name
            });
            llmLogger.response(analysisId, false, {
              status: error.status,
              message: error.data?.error || error.message
            });
            
            return (
              <Alert severity="error" sx={{ mb: 2 }}>
                <Typography variant="body2">
                  <strong>Error:</strong> {error.data?.error || error.message || 'Failed to generate explanation'}
                </Typography>
                {error.status && (
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                    Status: {error.status}
                  </Typography>
                )}
              </Alert>
            );
          })()}
          
          {isLoading && (
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, py: 3 }}>
              <CircularProgress size={32} />
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Generating AI explanation...
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  This may take a few seconds using LLaMA 3.1 70B
                </Typography>
              </Box>
            </Box>
          )}
          
          {explanation && !isLoading && (() => {
            const content = explanation.content || explanation.narrative_text;
            
            // Log successful explanation display
            if (content && content !== 'No explanation available') {
              explanationLogger.workflow(analysisId, 'Explanation displayed', {
                contentLength: content.length,
                method: explanation.method || method,
                confidence: explanation.confidence_score || confidence,
                hasTimestamp: !!timestamp
              });
              llmLogger.response(analysisId, true, {
                contentLength: content.length,
                method: explanation.method || method,
                generationTime: explanation.generation_time
              });
            } else {
              explanationLogger.warn('No explanation content available for display', {
                analysisId,
                explanation
              });
            }
            
            return (
              <Box>
                <Typography variant="body1" paragraph sx={{ lineHeight: 1.7 }}>
                  {content || 'No explanation available'}
                </Typography>
                
                {timestamp && (
                  <>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="caption" color="text.secondary">
                      Generated: {formatTimestamp(timestamp)}
                    </Typography>
                  </>
                )}
              </Box>
            );
          })()}
          
          {!explanation && !isLoading && !error && (
            <Box sx={{ textAlign: 'center', py: 3 }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                No explanation available
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Click "Generate" to create an AI explanation
              </Typography>
            </Box>
          )}
          
          {children && (
            <>
              <Divider sx={{ my: 2 }} />
              {children}
            </>
          )}
        </Collapse>
      </CardContent>
    </Card>
  );
};

export default ExplanationCard;