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
  variant = 'standard', // 'summary' (Standard), 'standard' (Enhanced), 'detailed' (Premium)
  confidence = null,
  method = null,
  modelName = null,
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

  // Handle detail level changes - only when user explicitly changes the dropdown
  const handleDetailLevelChange = (newDetailLevel) => {
    setSelectedDetail(newDetailLevel);
    if (onRefresh) {
      explanationLogger.workflow(analysisId, 'Detail level changed by user', { 
        from: selectedDetail,
        to: newDetailLevel 
      });
      onRefresh(newDetailLevel);
    }
  };

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

  const formatModelName = (modelName) => {
    if (!modelName) return 'LLM';
    
    // Convert common model names to user-friendly formats
    if (modelName.includes('phi3') || modelName.includes('Phi3')) return 'Phi3 3.8B';
    if (modelName.includes('llama3.1:8b') || modelName.includes('LLaMA-3.1-8B')) return 'LLaMA 3.1 8B';
    if (modelName.includes('llama3.1:70b') || modelName.includes('LLaMA-3.1-70B')) return 'LLaMA 3.1 70B';
    if (modelName.includes('llama') || modelName.includes('LLaMA')) return 'LLaMA';
    
    // Default to a cleaned up version of the model name
    return modelName.replace(/[_-]/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '';
    try {
      const date = new Date(timestamp);
      return date.toLocaleString();
    } catch {
      return '';
    }
  };

  const formatStructuredContent = (content) => {
    if (!content) return null;
    
    // For summary mode, keep formatting minimalistic - no visual enhancements
    if (selectedDetail === 'summary') {
      // Handle basic bold formatting but without enhanced styling
      if (content.includes('**')) {
        const parts = content.split(/(\*\*.*?\*\*)/g);
        return (
          <Typography variant="body1" paragraph sx={{ lineHeight: 1.7 }}>
            {parts.map((part, index) => {
              if (part.startsWith('**') && part.endsWith('**')) {
                return (
                  <Box 
                    key={index} 
                    component="span" 
                    sx={{ fontWeight: 600 }}
                  >
                    {part.slice(2, -2)}
                  </Box>
                );
              }
              return part;
            })}
          </Typography>
        );
      }
      // Return clean paragraph for summary
      return (
        <Typography variant="body1" paragraph sx={{ lineHeight: 1.7 }}>
          {content}
        </Typography>
      );
    }
    
    // Enhanced structured content parsing for standard and detailed modes
    if (content.includes('**') && content.includes(':')) {
      // Split by double line breaks first to get major sections
      const majorSections = content.split('\n\n').filter(section => section.trim());
      
      return (
        <Box sx={{ lineHeight: 1.8 }}>
          {majorSections.map((section, index) => {
            const trimmed = section.trim();
            // Pattern 1: **Title:** content (primary sections)  
            // Handle various malformed LLM outputs with enhanced regex patterns
            let primaryMatch = trimmed.match(/^\*\*(.*?):\*\*+\s*(.*)/s);
            
            // If no match, try standard format
            if (!primaryMatch) {
              primaryMatch = trimmed.match(/^\*\*(.*?):\*\*\s*(.*)/s);
            }
            
            // Try with missing space after colon
            if (!primaryMatch) {
              primaryMatch = trimmed.match(/^\*\*(.*?):\*\*([^*].*)/s);
            }
            
            // Try with wrong asterisk count (***Title:*** content)
            if (!primaryMatch) {
              primaryMatch = trimmed.match(/^\*{3,}(.*?):\*{3,}\s*(.*)/s);
            }
            
            // Try with extra asterisks in content (**Title:** ***content***)
            if (!primaryMatch) {
              primaryMatch = trimmed.match(/^\*\*(.*?):\*\*\s*\*+([^*].*?)\**/s);
            }
            
            // Pattern 2: **TITLE** content (fallback for missing colon)
            const fallbackMatch = !primaryMatch ? trimmed.match(/^\*\*([^*]+)\*\*\s*(.*)/s) : null;
            
            if (primaryMatch || fallbackMatch) {
              const match = primaryMatch || fallbackMatch;
              const title = match[1].trim();
              let content = match[2].trim();
              
              // If content is empty, check if it's in the next major section
              if (!content && index + 1 < majorSections.length) {
                const nextSection = majorSections[index + 1];
                if (nextSection && !nextSection.startsWith('**')) {
                  content = nextSection.trim();
                  majorSections.splice(index + 1, 1); // Remove the consumed section
                }
              }
              
              // Determine section type for enhanced styling
              const sectionType = getSectionType(title);
              
              return (
                <Box key={index} sx={{ mb: 2.5 }}>
                  <Typography 
                    variant="subtitle2" 
                    sx={{ 
                      fontWeight: 700, 
                      color: sectionType.color,
                      mb: 1,
                      fontSize: '0.95rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.5px'
                    }}
                  >
                    {sectionType.icon} {title}
                  </Typography>
                  <Typography 
                    variant="body2" 
                    sx={{ 
                      pl: 2, 
                      borderLeft: 3, 
                      borderColor: sectionType.borderColor,
                      backgroundColor: sectionType.bgColor,
                      p: 1.5,
                      borderRadius: 1,
                      fontSize: '0.9rem',
                      wordBreak: 'break-word',
                      overflowWrap: 'break-word',
                      whiteSpace: 'pre-wrap'
                    }}
                  >
                    {content}
                  </Typography>
                </Box>
              );
            }
            
            // Pattern 2: **Bold text** within paragraphs
            if (trimmed.includes('**') && !trimmed.match(/^\*\*.*\*\*:/)) {
              const parts = trimmed.split(/(\*\*.*?\*\*)/g);
              
              return (
                <Typography key={index} variant="body1" paragraph sx={{ mb: 2, lineHeight: 1.7 }}>
                  {parts.map((part, partIndex) => {
                    if (part.startsWith('**') && part.endsWith('**')) {
                      return (
                        <Box 
                          key={partIndex} 
                          component="span" 
                          sx={{ fontWeight: 600, color: 'primary.main' }}
                        >
                          {part.slice(2, -2)}
                        </Box>
                      );
                    }
                    return part;
                  })}
                </Typography>
              );
            }
            
            // Pattern 3: Regular paragraph - enhanced fallback styling for Premium mode
            if (selectedDetail === 'detailed' && trimmed.length > 50) {
              // For Premium mode, style longer paragraphs as premium section-like content
              // Try to auto-detect if this might be a section content
              const isPossibleSectionContent = trimmed.length > 100 || 
                                               /\b(analysis|assessment|strategy|outlook|recommendation)\b/i.test(trimmed);
              
              const sectionStyle = isPossibleSectionContent ? {
                color: 'success.main',
                borderColor: 'success.light', 
                bgColor: 'rgba(76, 175, 80, 0.08)',
                icon: '📈'
              } : {
                color: 'primary.main',
                borderColor: 'primary.light',
                bgColor: 'rgba(25, 118, 210, 0.08)', 
                icon: '📋'
              };
              
              return (
                <Box key={index} sx={{ mb: 2.5 }}>
                  <Typography 
                    variant="body2" 
                    sx={{ 
                      pl: 2, 
                      borderLeft: 3, 
                      borderColor: sectionStyle.borderColor,
                      backgroundColor: sectionStyle.bgColor,
                      p: 1.5,
                      borderRadius: 1,
                      fontSize: '0.9rem',
                      lineHeight: 1.7,
                      wordBreak: 'break-word',
                      overflowWrap: 'break-word',
                      whiteSpace: 'pre-wrap'
                    }}
                  >
                    {trimmed}
                  </Typography>
                </Box>
              );
            }
            
            // Regular paragraph for other modes
            return (
              <Typography key={index} variant="body1" paragraph sx={{ mb: 2, lineHeight: 1.7 }}>
                {trimmed}
              </Typography>
            );
          })}
        </Box>
      );
    }
    
    // Fallback: Check for inline bold formatting
    if (content.includes('**')) {
      const parts = content.split(/(\*\*.*?\*\*)/g);
      
      return (
        <Typography variant="body1" paragraph sx={{ lineHeight: 1.7 }}>
          {parts.map((part, index) => {
            if (part.startsWith('**') && part.endsWith('**')) {
              return (
                <Box 
                  key={index} 
                  component="span" 
                  sx={{ fontWeight: 600, color: 'primary.main' }}
                >
                  {part.slice(2, -2)}
                </Box>
              );
            }
            return part;
          })}
        </Typography>
      );
    }
    
    // Final fallback to regular paragraph
    return (
      <Typography variant="body1" paragraph sx={{ lineHeight: 1.7 }}>
        {content}
      </Typography>
    );
  };

  // Helper function to determine section styling based on title
  const getSectionType = (title) => {
    const titleLower = title.toLowerCase();
    
    if (titleLower.includes('investment') || titleLower.includes('recommendation') || titleLower.includes('thesis')) {
      return {
        color: 'success.main',
        borderColor: 'success.light',
        bgColor: 'rgba(76, 175, 80, 0.08)',
        icon: '📈'
      };
    } else if (titleLower.includes('technical') || titleLower.includes('indicator')) {
      return {
        color: 'info.main',
        borderColor: 'info.light',
        bgColor: 'rgba(33, 150, 243, 0.08)',
        icon: '📊'
      };
    } else if (titleLower.includes('risk') || titleLower.includes('challenge')) {
      return {
        color: 'warning.main',
        borderColor: 'warning.light',
        bgColor: 'rgba(255, 152, 0, 0.08)',
        icon: '⚠️'
      };
    } else if (titleLower.includes('market') || titleLower.includes('outlook') || titleLower.includes('context')) {
      return {
        color: 'secondary.main',
        borderColor: 'secondary.light',
        bgColor: 'rgba(156, 39, 176, 0.08)',
        icon: '🏢'
      };
    } else if (titleLower.includes('sentiment')) {
      return {
        color: '#9c27b0',
        borderColor: '#ba68c8',
        bgColor: 'rgba(156, 39, 176, 0.08)',
        icon: '💭'
      };
    }
    
    // Default styling
    return {
      color: 'primary.main',
      borderColor: 'primary.light',
      bgColor: 'rgba(25, 118, 210, 0.08)',
      icon: '📋'
    };
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
            {method === 'llama' && modelName && (
              <Tooltip title={`Generated by ${formatModelName(modelName)} model`}>
                <Chip 
                  icon={getMethodIcon(method)}
                  label={formatModelName(modelName)}
                  size="small"
                  color="primary"
                  variant="outlined"
                />
              </Tooltip>
            )}
            {method && method !== 'llama' && (
              <Tooltip title="Template Generated - LLM Unavailable">
                <Chip 
                  icon={getMethodIcon(method)}
                  label="Template"
                  size="small"
                  color="default"
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
                    onChange={(e) => handleDetailLevelChange(e.target.value)}
                    disabled={isLoading}
                  >
                    <MenuItem value="summary">Standard</MenuItem>
                    <MenuItem value="standard">Enhanced</MenuItem>
                    <MenuItem value="detailed">Premium</MenuItem>
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
                  This may take a few seconds using AI model
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
                {formatStructuredContent(content) || (
                  <Typography variant="body1" paragraph sx={{ lineHeight: 1.7 }}>
                    No explanation available
                  </Typography>
                )}
                
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