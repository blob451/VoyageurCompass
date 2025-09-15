"""
Multilingual prompt templates for financial explanations.
Provides localised prompt templates for French and Spanish to ensure culturally appropriate
and linguistically accurate financial analysis explanations.
"""

from typing import Dict, Any, Optional


class MultilingualPromptTemplates:
    """Templates for generating multilingual financial explanations."""
    
    def __init__(self):
        # Financial terminology translations
        self.financial_terminology = {
            "fr": {
                "buy": "ACHAT",
                "sell": "VENTE", 
                "hold": "CONSERVATION",
                "technical_analysis": "analyse technique",
                "investment_thesis": "thèse d'investissement",
                "risk_analysis": "analyse des risques",
                "market_context": "contexte de marché",
                "bullish": "haussier",
                "bearish": "baissier",
                "neutral": "neutre",
                "strong": "forte",
                "moderate": "modérée",
                "weak": "faible",
                "recommendation": "recommandation",
                "confidence": "confiance",
                "indicators": "indicateurs",
                "momentum": "momentum",
                "volatility": "volatilité",
                "resistance": "résistance",
                "support": "support"
            },
            "es": {
                "buy": "COMPRA",
                "sell": "VENTA",
                "hold": "MANTENER",
                "technical_analysis": "análisis técnico",
                "investment_thesis": "tesis de inversión", 
                "risk_analysis": "análisis de riesgo",
                "market_context": "contexto de mercado",
                "bullish": "alcista",
                "bearish": "bajista", 
                "neutral": "neutral",
                "strong": "fuerte",
                "moderate": "moderada",
                "weak": "débil",
                "recommendation": "recomendación",
                "confidence": "confianza",
                "indicators": "indicadores",
                "momentum": "momentum",
                "volatility": "volatilidad",
                "resistance": "resistencia",
                "support": "soporte"
            }
        }
        
        # Number and currency formatting rules
        self.number_formatting = {
            "fr": {
                "decimal_separator": ",",
                "thousands_separator": " ",
                "currency_symbol": "€",
                "currency_position": "after",  # 10,50 €
                "percentage_format": "10,5 %"
            },
            "es": {
                "decimal_separator": ",", 
                "thousands_separator": ".",
                "currency_symbol": "€",
                "currency_position": "after",  # 10,50 €
                "percentage_format": "10,5%"
            }
        }
    
    def get_translation_prompt_template(self, target_language: str, detail_level: str) -> str:
        """
        Get translation prompt template for specific language and detail level.
        
        Args:
            target_language: Target language code ('fr' or 'es')
            detail_level: Detail level ('summary', 'standard', 'detailed')
            
        Returns:
            Formatted prompt template string
        """
        if target_language == "fr":
            return self._get_french_translation_template(detail_level)
        elif target_language == "es":
            return self._get_spanish_translation_template(detail_level)
        else:
            raise ValueError(f"Unsupported language: {target_language}")
    
    def _get_french_translation_template(self, detail_level: str) -> str:
        """Get French translation prompt template."""
        terminology = self.financial_terminology["fr"]
        
        base_template = f"""Vous êtes un traducteur financier professionnel spécialisé dans l'analyse d'investissement.

Terminologie financière clé (utiliser exactement ces termes) :
- BUY → {terminology["buy"]}
- SELL → {terminology["sell"]} 
- HOLD → {terminology["hold"]}
- Technical Analysis → {terminology["technical_analysis"]}
- Bullish → {terminology["bullish"]}
- Bearish → {terminology["bearish"]}
- Strong → {terminology["strong"]}
- Moderate → {terminology["moderate"]}
- Weak → {terminology["weak"]}

Règles de traduction :
1. Préservez exactement tous les symboles d'actions (ex: AAPL, MSFT)
2. Conservez tous les chiffres, pourcentages et ratios exactement comme ils apparaissent
3. Utilisez la terminologie financière française standard
4. Maintenez le ton professionnel d'analyse d'investissement
5. Adaptez la structure des phrases au français naturel
6. Utilisez la virgule comme séparateur décimal (10,5% au lieu de 10.5%)

Format de sortie attendu : Texte français uniquement, sans explications additionnelles.

Texte anglais à traduire :
{{english_text}}

Traduction française :"""
        
        return base_template
    
    def _get_spanish_translation_template(self, detail_level: str) -> str:
        """Get Spanish translation prompt template.""" 
        terminology = self.financial_terminology["es"]
        
        base_template = f"""Eres un traductor financiero profesional especializado en análisis de inversión.

Terminología financiera clave (usar exactamente estos términos):
- BUY → {terminology["buy"]}
- SELL → {terminology["sell"]}
- HOLD → {terminology["hold"]}
- Technical Analysis → {terminology["technical_analysis"]}
- Bullish → {terminology["bullish"]}
- Bearish → {terminology["bearish"]}
- Strong → {terminology["strong"]}
- Moderate → {terminology["moderate"]}
- Weak → {terminology["weak"]}

Reglas de traducción :
1. Preserva exactamente todos los símbolos de acciones (ej: AAPL, MSFT)
2. Conserva todos los números, porcentajes y ratios exactamente como aparecen
3. Usa la terminología financiera española estándar
4. Mantén el tono profesional de análisis de inversión
5. Adapta la estructura de las frases al español natural
6. Usa la coma como separador decimal (10,5% en lugar de 10.5%)

Formato de salida esperado: Texto en español únicamente, sin explicaciones adicionales.

Texto inglés a traducir:
{{english_text}}

Traducción española:"""
        
        return base_template
    
    def get_direct_generation_prompt(
        self,
        language: str,
        analysis_data: Dict[str, Any],
        detail_level: str = "standard"
    ) -> str:
        """
        Get prompt for direct generation in target language (future enhancement).
        
        Args:
            language: Target language code
            analysis_data: Analysis data dictionary
            detail_level: Detail level for explanation
            
        Returns:
            Formatted prompt for direct generation in target language
        """
        if language == "fr":
            return self._get_french_generation_prompt(analysis_data, detail_level)
        elif language == "es":
            return self._get_spanish_generation_prompt(analysis_data, detail_level)
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    def _get_french_generation_prompt(self, analysis_data: Dict[str, Any], detail_level: str) -> str:
        """Get French direct generation prompt."""
        symbol = analysis_data.get("symbol", "UNKNOWN")
        score = analysis_data.get("score_0_10", 0)
        terminology = self.financial_terminology["fr"]
        
        # Determine recommendation in French
        if score >= 7:
            expected_rec = terminology["buy"]
            score_desc = f"{terminology['strong']} signal {terminology['bullish']}"
        elif score >= 4:
            expected_rec = terminology["hold"]
            score_desc = f"signaux mixtes/{terminology['neutral']}"
        else:
            expected_rec = terminology["sell"]
            score_desc = f"{terminology['weak']} signal {terminology['bearish']}"
        
        if detail_level == "summary":
            prompt = f"""Analyse financière : {symbol} obtient une note de {score}/10, indiquant des {score_desc}.

IMPORTANT : Basé sur la note de {score}/10, votre {terminology["recommendation"]} DOIT être {expected_rec}.
- Notes 7-10 = {terminology["buy"]}
- Notes 4-6,9 = {terminology["hold"]}  
- Notes 0-3,9 = {terminology["sell"]}

Fournissez une {terminology["recommendation"]} {expected_rec} claire et conversationnelle en 50-60 mots. Utilisez un format de paragraphe simple sans en-têtes de section ni formatage. Soyez direct, convivial et concis avec des phrases complètes."""
        
        elif detail_level == "detailed":
            prompt = f"""Analyse financière : {symbol} obtient une note de {score}/10 basée sur les {terminology["indicators"]} techniques, indiquant des {score_desc}.

IMPORTANT : Basé sur la note de {score}/10, votre {terminology["recommendation"]} DOIT être {expected_rec}.
- Notes 7-10 = {terminology["buy"]}
- Notes 4-6,9 = {terminology["hold"]}
- Notes 0-3,9 = {terminology["sell"]}

Fournissez une analyse {expected_rec} complète pour {symbol} en 250-300 mots en utilisant cette structure :

**{terminology["investment_thesis"].title()} :** {terminology["recommendation"].title()} {expected_rec} claire avec niveau de {terminology["confidence"]} et raisonnement principal

**{terminology["indicators"].title()} Techniques :** Analyse détaillée des {terminology["indicators"]} clés soutenant la décision {expected_rec}

**{terminology["risk_analysis"].title()} :** Principaux risques, défis et stratégies d'atténuation des risques

**{terminology["market_context"].title()} :** Perspectives de prix, catalyseurs et facteurs d'environnement de marché

Utilisez le langage de recherche d'investissement professionnel. Assurez-vous que toutes les sections soutiennent la {terminology["recommendation"]} {expected_rec}."""
        
        else:  # standard
            prompt = f"""Analyse d'investissement : {symbol} reçoit une note de {score}/10 de l'{terminology["technical_analysis"]}, indiquant des {score_desc}.

IMPORTANT : Basé sur la note de {score}/10, votre {terminology["recommendation"]} DOIT être {expected_rec}.
- Notes 7-10 = {terminology["buy"]}
- Notes 4-6,9 = {terminology["hold"]}
- Notes 0-3,9 = {terminology["sell"]}

Fournissez une {terminology["recommendation"]} {expected_rec} professionnelle pour {symbol} en 100-120 mots incluant :
- Décision {expected_rec} claire avec niveau de {terminology["confidence"]}
- 2-3 facteurs techniques clés de soutien
- Considération principale du risque et bref aperçu du marché

Utilisez un langage d'investissement professionnel avec des phrases complètes."""
        
        return prompt
    
    def _get_spanish_generation_prompt(self, analysis_data: Dict[str, Any], detail_level: str) -> str:
        """Get Spanish direct generation prompt."""
        symbol = analysis_data.get("symbol", "UNKNOWN")
        score = analysis_data.get("score_0_10", 0)
        terminology = self.financial_terminology["es"]
        
        # Determine recommendation in Spanish
        if score >= 7:
            expected_rec = terminology["buy"]
            score_desc = f"señal {terminology['strong']} {terminology['bullish']}"
        elif score >= 4:
            expected_rec = terminology["hold"]
            score_desc = f"señales mixtas/{terminology['neutral']}"
        else:
            expected_rec = terminology["sell"]
            score_desc = f"señal {terminology['weak']} {terminology['bearish']}"
        
        if detail_level == "summary":
            prompt = f"""Análisis financiero: {symbol} obtiene una puntuación de {score}/10, indicando {score_desc}.

IMPORTANTE: Basado en la puntuación de {score}/10, tu {terminology["recommendation"]} DEBE ser {expected_rec}.
- Puntuaciones 7-10 = {terminology["buy"]}
- Puntuaciones 4-6,9 = {terminology["hold"]}
- Puntuaciones 0-3,9 = {terminology["sell"]}

Proporciona una {terminology["recommendation"]} {expected_rec} clara y conversacional en 50-60 palabras. Usa formato de párrafo simple sin encabezados de sección o formateo. Sé directo, amigable y conciso con oraciones completas."""
        
        elif detail_level == "detailed":
            prompt = f"""Análisis financiero: {symbol} obtiene una puntuación de {score}/10 basada en {terminology["indicators"]} técnicos, indicando {score_desc}.

IMPORTANTE: Basado en la puntuación de {score}/10, tu {terminology["recommendation"]} DEBE ser {expected_rec}.
- Puntuaciones 7-10 = {terminology["buy"]}
- Puntuaciones 4-6,9 = {terminology["hold"]}
- Puntuaciones 0-3,9 = {terminology["sell"]}

Proporciona un análisis {expected_rec} completo para {symbol} en 250-300 palabras usando esta estructura:

**{terminology["investment_thesis"].title()}:** {terminology["recommendation"].title()} {expected_rec} clara con nivel de {terminology["confidence"]} y razonamiento principal

**{terminology["indicators"].title()} Técnicos:** Análisis detallado de {terminology["indicators"]} clave que apoyan la decisión {expected_rec}

**{terminology["risk_analysis"].title()}:** Principales riesgos, desafíos y estrategias de mitigación de riesgos

**{terminology["market_context"].title()}:** Perspectivas de precio, catalizadores y factores del entorno de mercado

Usa lenguaje de investigación de inversión profesional. Asegúrate de que todas las secciones apoyen la {terminology["recommendation"]} {expected_rec}."""
        
        else:  # standard
            prompt = f"""Análisis de inversión: {symbol} recibe una puntuación de {score}/10 del {terminology["technical_analysis"]}, indicando {score_desc}.

IMPORTANTE: Basado en la puntuación de {score}/10, tu {terminology["recommendation"]} DEBE ser {expected_rec}.
- Puntuaciones 7-10 = {terminology["buy"]}
- Puntuaciones 4-6,9 = {terminology["hold"]}
- Puntuaciones 0-3,9 = {terminology["sell"]}

Proporciona una {terminology["recommendation"]} {expected_rec} profesional para {symbol} en 100-120 palabras incluyendo:
- Decisión {expected_rec} clara con nivel de {terminology["confidence"]}
- 2-3 factores técnicos clave de apoyo
- Consideración principal de riesgo y breve perspectiva del mercado

Usa lenguaje de inversión profesional con oraciones completas."""
        
        return prompt
    
    def format_numbers_for_language(self, text: str, language: str) -> str:
        """
        Format numbers according to language conventions.
        
        Args:
            text: Text containing numbers to format
            language: Target language code
            
        Returns:
            Text with properly formatted numbers
        """
        if language not in self.number_formatting:
            return text
        
        import re
        formatting = self.number_formatting[language]
        
        try:
            # Format percentages (10.5% -> 10,5%)
            text = re.sub(r'(\d+)\.(\d+)%', rf'\1{formatting["decimal_separator"]}\2%', text)
            
            # Format decimal numbers (10.50 -> 10,50)
            text = re.sub(r'(\d+)\.(\d+)', rf'\1{formatting["decimal_separator"]}\2', text)
            
            return text
        except Exception:
            # Return original text if formatting fails
            return text
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages with their full names."""
        return {
            "fr": "Français",
            "es": "Español",
            "en": "English"
        }
    
    def get_terminology_for_language(self, language: str) -> Dict[str, str]:
        """Get financial terminology dictionary for a language."""
        return self.financial_terminology.get(language, {})


# Singleton instance
_prompt_templates = None


def get_multilingual_templates() -> MultilingualPromptTemplates:
    """Get singleton instance of MultilingualPromptTemplates."""
    global _prompt_templates
    if _prompt_templates is None:
        _prompt_templates = MultilingualPromptTemplates()
    return _prompt_templates