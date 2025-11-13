#!/usr/bin/env python3
"""
AI-Powered Biological Cascade Effect Predictor
- Uses Gemini API to predict downstream effects when drugs act on targets
- Stores predictions in Neo4j graph database
- Supports multi-hop cascade prediction (1-3 levels deep)
- Provides confidence scoring and reasoning for each prediction
"""

import google.generativeai as genai
import json
import logging
import time
from datetime import datetime
from neo4j import GraphDatabase
from typing import Dict, Optional, List, Any, Tuple
import os
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CascadeEffect:
    """Data class for a single cascade effect prediction"""
    entity_name: str          # Name of the affected entity
    entity_type: str          # Pathway, Gene, Metabolite, CellularProcess, or Protein
    effect_type: str          # inhibits, activates, upregulates, downregulates, modulates
    confidence: float         # 0.0 to 1.0
    reasoning: str            # AI's explanation
    depth: int                # 1, 2, or 3 (hops from original target)
    source_entity: str        # What entity causes this effect
    additional_info: Optional[Dict] = None  # Optional extra data

@dataclass
class CascadePrediction:
    """Complete cascade prediction result"""
    drug_name: str
    target_name: str
    direct_effects: List[CascadeEffect]      # 1-hop effects
    secondary_effects: List[CascadeEffect]   # 2-hop effects
    tertiary_effects: List[CascadeEffect]    # 3-hop effects (optional)
    prediction_timestamp: str
    prediction_source: str
    total_confidence: float  # Average confidence of all predictions


class BiologicalCascadePredictor:
    """
    AI-powered predictor for biological cascade effects
    """
    
    def __init__(self, gemini_api_key: str = None, neo4j_uri: str = None,
                 neo4j_user: str = None, neo4j_password: str = None, 
                 neo4j_database: str = "neo4j"):
        """
        Initialize the cascade predictor
        
        Args:
            gemini_api_key: Google Gemini API key
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name
        """
        self.gemini_model = None
        self.driver = None
        self.database = neo4j_database
        
        # Initialize Gemini API
        if gemini_api_key:
            self._initialize_gemini(gemini_api_key)
        else:
            logger.warning("No Gemini API key provided - cascade prediction will be disabled")
        
        # Initialize Neo4j connection
        if neo4j_uri and neo4j_user and neo4j_password:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self.database = neo4j_database
            logger.info("Neo4j connection initialized")
        else:
            logger.warning("No Neo4j credentials provided - database operations will be disabled")
    
    def _initialize_gemini(self, api_key: str):
        """Initialize Gemini API with best available model"""
        genai.configure(api_key=api_key)
        
        # Try to find the best available model (prioritize Flash models for speed)
        model_names = [
            'models/gemini-2.0-flash-exp',
            'models/gemini-1.5-flash-latest',
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro',
            'models/gemini-pro'
        ]
        
        for model_name in model_names:
            try:
                self.gemini_model = genai.GenerativeModel(model_name)
                # Test the model
                test_response = self.gemini_model.generate_content(
                    "Respond with 'OK'",
                    generation_config={'temperature': 0.3}
                )
                logger.info(f"Gemini API initialized successfully with model: {model_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to initialize model {model_name}: {str(e)[:100]}")
                continue
        
        if not self.gemini_model:
            logger.error("Failed to initialize any Gemini model")
    
    def create_cascade_schema(self):
        """Create Neo4j schema for cascade predictions (one-time setup)"""
        if not self.driver:
            logger.error("No Neo4j connection available")
            return False
        
        logger.info("Creating cascade prediction schema...")
        
        try:
            with self.driver.session(database=self.database) as session:
                # Create constraints for new node types
                session.run("""
                    CREATE CONSTRAINT pathway_name IF NOT EXISTS 
                    FOR (p:Pathway) REQUIRE p.name IS UNIQUE
                """)
                
                session.run("""
                    CREATE CONSTRAINT gene_symbol IF NOT EXISTS 
                    FOR (g:Gene) REQUIRE g.symbol IS UNIQUE
                """)
                
                session.run("""
                    CREATE CONSTRAINT metabolite_name IF NOT EXISTS 
                    FOR (m:Metabolite) REQUIRE m.name IS UNIQUE
                """)
                
                session.run("""
                    CREATE CONSTRAINT process_name IF NOT EXISTS 
                    FOR (cp:CellularProcess) REQUIRE cp.name IS UNIQUE
                """)
                
                # Create indexes for performance
                session.run("""
                    CREATE INDEX pathway_category IF NOT EXISTS 
                    FOR (p:Pathway) ON (p.category)
                """)
                
                session.run("""
                    CREATE INDEX gene_name IF NOT EXISTS 
                    FOR (g:Gene) ON (g.name)
                """)
                
                logger.info("Cascade schema created successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error creating cascade schema: {e}")
            return False
    
    def predict_cascade_effects(self, drug_name: str, target_name: str, 
                               depth: int = 2, 
                               additional_context: str = "") -> Optional[CascadePrediction]:
        """
        Predict cascade effects using Gemini API
        
        Args:
            drug_name: Name of the drug
            target_name: Name of the primary target
            depth: How many hops to predict (1, 2, or 3)
            additional_context: Additional information about the drug-target interaction
        
        Returns:
            CascadePrediction object or None if prediction fails
        """
        if not self.gemini_model:
            logger.error("Gemini API not available")
            return None
        
        # Construct the prompt
        prompt = self._build_cascade_prompt(drug_name, target_name, depth, additional_context)
        
        logger.info(f"Predicting cascade effects for {drug_name} → {target_name} (depth: {depth})")
        
        try:
            # Query Gemini API with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.gemini_model.generate_content(
                        prompt,
                        generation_config={
                            'temperature': 0.3,  # Lower temperature for more consistent results
                            'top_p': 0.8,
                            'top_k': 40,
                            'max_output_tokens': 2048,
                        }
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed, retrying... Error: {str(e)[:100]}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        raise
            
            if not response.text:
                logger.error("Empty response from Gemini API")
                return None
            
            # Parse the response
            cascade_data = self._parse_cascade_response(response.text, drug_name, target_name)
            
            if cascade_data:
                logger.info(f"Successfully predicted cascade with {len(cascade_data.direct_effects)} direct effects")
            
            return cascade_data
            
        except Exception as e:
            logger.error(f"Error during cascade prediction: {e}")
            return None
    
    def _build_cascade_prompt(self, drug_name: str, target_name: str,
                             depth: int, additional_context: str) -> str:
        """Build the prompt for Gemini API"""

        depth_instructions = {
            1: "Focus only on DIRECT effects (1-hop from the target).",
            2: "Include both DIRECT effects (1-hop) and SECONDARY effects (2-hop).",
            3: "Include DIRECT effects (1-hop), SECONDARY effects (2-hop), and TERTIARY effects (3-hop)."
        }

        json_example_template = """{
  "direct_effects": [
    {
      "entity_name": "Prostaglandin synthesis pathway",
      "entity_type": "Pathway",
      "effect_type": "inhibits",
      "confidence": 0.95,
      "reasoning": "COX-2 is the rate-limiting enzyme in prostaglandin production",
      "source_entity": "{target_name}"
    }
  ],
  "secondary_effects": [
    {
      "entity_name": "Inflammatory response",
      "entity_type": "CellularProcess",
      "effect_type": "downregulates",
      "confidence": 0.88,
      "reasoning": "Reduced prostaglandins lead to decreased inflammation",
      "source_entity": "Prostaglandin synthesis pathway"
    }
  ],
  "tertiary_effects": []
}"""

        json_example = json_example_template.replace("{target_name}", target_name)
        depth_text = depth_instructions.get(depth, depth_instructions[2])
        context_text = additional_context if additional_context else "Standard pharmacological interaction"

        prompt = f"""You are an expert pharmacologist and systems biologist.

CRITICAL VALIDATION STEP: Before predicting cascade effects, you MUST verify that "{drug_name}" actually directly targets "{target_name}".

If "{drug_name}" does NOT directly target "{target_name}" (e.g., it's an indirect relationship, crosstalk, or hypothetical), you MUST:
1. In the FIRST effect's reasoning, explicitly state: "IMPORTANT: The drug {drug_name} does NOT directly target {target_name}. This cascade is hypothetical and based on indirect mechanisms or crosstalk."
2. Include this disclaimer in your reasoning for at least the first direct effect
3. Use lower confidence scores (<0.6) for all effects
4. Clearly distinguish between direct vs indirect/hypothetical relationships in your reasoning

If the relationship is completely unsupported by literature, return an empty cascade or explain why this relationship is unlikely.

Analyze the biological cascade effects when drug "{drug_name}" acts on target "{target_name}".

{depth_text}

Context: {context_text}

VALIDATION RULES:
- If the relationship is NOT direct, the FIRST effect MUST include a clear disclaimer in its reasoning
- If uncertain about the drug-target relationship, reduce all confidence scores by 0.3 and state uncertainty in reasoning
- If the relationship is indirect (e.g., crosstalk, secondary effects), explicitly mention this in the reasoning
- Only predict cascades for relationships that have some biological plausibility, even if indirect

Predict downstream effects on:
- Biological pathways
- Gene expression
- Metabolites
- Cellular processes
- Other proteins

For each effect, specify:
1. Entity name (be specific)
2. Entity type (Pathway, Gene, Metabolite, CellularProcess, or Protein)
3. Effect type (inhibits, activates, upregulates, downregulates, or modulates)
4. Confidence score (0.0 to 1.0)
5. Brief reasoning

RESPOND ONLY WITH VALID JSON in this EXACT format:

{json_example}

Focus on high-confidence, well-established biological relationships.
Provide 3-5 direct effects and 2-4 secondary effects.
Be specific with pathway and gene names.

JSON Response:"""

        return prompt
    
    def _parse_cascade_response(self, response_text: str, drug_name: str, 
                                target_name: str) -> Optional[CascadePrediction]:
        """Parse Gemini API response into CascadePrediction object"""
        
        try:
            # Clean up the response text
            response_text = response_text.strip()
            
            # Remove markdown formatting if present
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            # Extract JSON object
            if "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                response_text = response_text[json_start:json_end]
            
            # Parse JSON
            data = json.loads(response_text)
            
            # Convert to CascadeEffect objects
            direct_effects = [
                CascadeEffect(
                    entity_name=e['entity_name'],
                    entity_type=e['entity_type'],
                    effect_type=e['effect_type'],
                    confidence=float(e['confidence']),
                    reasoning=e['reasoning'],
                    depth=1,
                    source_entity=e.get('source_entity', target_name)
                )
                for e in data.get('direct_effects', [])
            ]
            
            secondary_effects = [
                CascadeEffect(
                    entity_name=e['entity_name'],
                    entity_type=e['entity_type'],
                    effect_type=e['effect_type'],
                    confidence=float(e['confidence']),
                    reasoning=e['reasoning'],
                    depth=2,
                    source_entity=e.get('source_entity', 'unknown')
                )
                for e in data.get('secondary_effects', [])
            ]
            
            tertiary_effects = [
                CascadeEffect(
                    entity_name=e['entity_name'],
                    entity_type=e['entity_type'],
                    effect_type=e['effect_type'],
                    confidence=float(e['confidence']),
                    reasoning=e['reasoning'],
                    depth=3,
                    source_entity=e.get('source_entity', 'unknown')
                )
                for e in data.get('tertiary_effects', [])
            ]
            
            # Calculate average confidence
            all_effects = direct_effects + secondary_effects + tertiary_effects
            avg_confidence = sum(e.confidence for e in all_effects) / len(all_effects) if all_effects else 0.0
            
            # Create CascadePrediction object
            prediction = CascadePrediction(
                drug_name=drug_name,
                target_name=target_name,
                direct_effects=direct_effects,
                secondary_effects=secondary_effects,
                tertiary_effects=tertiary_effects,
                prediction_timestamp=datetime.now().isoformat(),
                prediction_source="Gemini_API",
                total_confidence=avg_confidence
            )
            
            return prediction
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text[:500]}")
            return None
        except Exception as e:
            logger.error(f"Error parsing cascade response: {e}")
            return None
    
    def store_cascade_in_neo4j(self, cascade: CascadePrediction) -> bool:
        """
        Store cascade prediction in Neo4j database
        
        Args:
            cascade: CascadePrediction object to store
        
        Returns:
            True if successful, False otherwise
        """
        if not self.driver:
            logger.error("No Neo4j connection available")
            return False
        
        try:
            with self.driver.session(database=self.database) as session:
                # Process all effects
                all_effects = (cascade.direct_effects + cascade.secondary_effects + 
                             cascade.tertiary_effects)
                
                for effect in all_effects:
                    self._store_single_effect(session, cascade.target_name, effect, cascade)
                
                logger.info(f"Successfully stored cascade with {len(all_effects)} effects in Neo4j")
                return True
                
        except Exception as e:
            logger.error(f"Error storing cascade in Neo4j: {e}")
            return False
    
    def _store_single_effect(self, session, target_name: str, effect: CascadeEffect, 
                            cascade: CascadePrediction):
        """Store a single cascade effect in Neo4j"""
        
        # Create the affected entity node (using MERGE to avoid duplicates)
        entity_label = effect.entity_type
        
        # Create node based on entity type
        if entity_label == "Gene":
            session.run(f"""
                MERGE (e:Gene {{symbol: $entity_name}})
                ON CREATE SET e.name = $entity_name,
                             e.created_date = datetime()
            """, entity_name=effect.entity_name)
        else:
            session.run(f"""
                MERGE (e:{entity_label} {{name: $entity_name}})
                ON CREATE SET e.created_date = datetime()
            """, entity_name=effect.entity_name)
        
        # Create relationship from target to affected entity
        # Use different property names based on entity type (symbol vs name)
        property_name = "symbol" if entity_label == "Gene" else "name"
        
        session.run(f"""
            MATCH (t:Target {{name: $target_name}})
            MATCH (e:{entity_label} {{{property_name}: $entity_name}})
            MERGE (t)-[r:AFFECTS_DOWNSTREAM]->(e)
            SET r.effect_type = $effect_type,
                r.confidence = $confidence,
                r.reasoning = $reasoning,
                r.depth = $depth,
                r.source_entity = $source_entity,
                r.predicted_by = $predicted_by,
                r.prediction_date = datetime($prediction_date),
                r.drug_context = $drug_name,
                r.validated = false
        """,
            target_name=target_name,
            entity_name=effect.entity_name,
            effect_type=effect.effect_type,
            confidence=effect.confidence,
            reasoning=effect.reasoning,
            depth=effect.depth,
            source_entity=effect.source_entity,
            predicted_by=cascade.prediction_source,
            prediction_date=cascade.prediction_timestamp,
            drug_name=cascade.drug_name
        )
    
    def get_existing_cascade(self, drug_name: str, target_name: str, 
                            min_confidence: float = 0.0) -> Optional[CascadePrediction]:
        """
        Retrieve existing cascade prediction from Neo4j
        
        Args:
            drug_name: Drug name
            target_name: Target name
            min_confidence: Minimum confidence threshold
        
        Returns:
            CascadePrediction object or None if not found
        """
        if not self.driver:
            return None
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (t:Target {name: $target_name})-[r:AFFECTS_DOWNSTREAM]->(e)
                    WHERE r.drug_context = $drug_name AND r.confidence >= $min_confidence
                    RETURN e, labels(e)[0] as entity_type, r
                    ORDER BY r.depth, r.confidence DESC
                """, target_name=target_name, drug_name=drug_name, min_confidence=min_confidence)
                
                records = list(result)
                
                if not records:
                    return None
                
                # Group effects by depth
                direct_effects = []
                secondary_effects = []
                tertiary_effects = []
                
                for record in records:
                    entity_node = record['e']
                    entity_type = record['entity_type']
                    rel = record['r']
                    
                    # Get entity name (handle Gene's symbol vs other nodes' name)
                    entity_name = entity_node.get('symbol') if entity_type == 'Gene' else entity_node.get('name')
                    
                    effect = CascadeEffect(
                        entity_name=entity_name,
                        entity_type=entity_type,
                        effect_type=rel['effect_type'],
                        confidence=rel['confidence'],
                        reasoning=rel['reasoning'],
                        depth=rel['depth'],
                        source_entity=rel['source_entity']
                    )
                    
                    if effect.depth == 1:
                        direct_effects.append(effect)
                    elif effect.depth == 2:
                        secondary_effects.append(effect)
                    elif effect.depth == 3:
                        tertiary_effects.append(effect)
                
                # Calculate average confidence
                all_effects = direct_effects + secondary_effects + tertiary_effects
                avg_confidence = sum(e.confidence for e in all_effects) / len(all_effects) if all_effects else 0.0
                
                # Get prediction timestamp from first record
                prediction_date = records[0]['r'].get('prediction_date', datetime.now()).isoformat()
                
                cascade = CascadePrediction(
                    drug_name=drug_name,
                    target_name=target_name,
                    direct_effects=direct_effects,
                    secondary_effects=secondary_effects,
                    tertiary_effects=tertiary_effects,
                    prediction_timestamp=prediction_date,
                    prediction_source="Neo4j_Database",
                    total_confidence=avg_confidence
                )
                
                logger.info(f"Retrieved existing cascade with {len(all_effects)} effects from Neo4j")
                return cascade
                
        except Exception as e:
            logger.error(f"Error retrieving cascade from Neo4j: {e}")
            return None
    
    def _validate_drug_target_relationship(self, drug_name: str, target_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that the drug-target relationship exists in the database
        
        Returns:
            (exists, mechanism) - tuple of (bool, mechanism_description or None)
        """
        if not self.driver:
            return (True, None)  # Skip validation if no database connection
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (d:Drug {name: $drug_name})-[r:TARGETS]->(t:Target {name: $target_name})
                    RETURN r.mechanism as mechanism, r.relationship_type as relationship_type,
                           r.confidence as confidence, r.classified as classified
                """, drug_name=drug_name, target_name=target_name).single()
                
                if result:
                    mechanism = result.get('mechanism', 'Unknown mechanism')
                    relationship_type = result.get('relationship_type', 'Unknown')
                    classified = result.get('classified', False)
                    
                    mechanism_info = f"{relationship_type} - {mechanism}" if mechanism else relationship_type
                    return (True, mechanism_info if classified else None)
                else:
                    return (False, None)
        except Exception as e:
            logger.warning(f"Error validating drug-target relationship: {e}")
            return (True, None)  # Continue if validation fails (don't block prediction)
    
    def predict_and_store(self, drug_name: str, target_name: str, depth: int = 2,
                         force_repredict: bool = False, 
                         additional_context: str = "") -> Tuple[Optional[CascadePrediction], Optional[str]]:
        """
        Complete workflow: validate relationship, check existing, predict if needed, store results
        
        Args:
            drug_name: Drug name
            target_name: Target name
            depth: Prediction depth (1-3)
            force_repredict: Force new prediction even if exists
            additional_context: Additional context for prediction
        
        Returns:
            (CascadePrediction object or None, warning message or None)
        """
        # Validate drug-target relationship exists
        relationship_exists, mechanism = self._validate_drug_target_relationship(drug_name, target_name)
        warning = None
        
        if not relationship_exists:
            warning = f"⚠️ WARNING: The drug-target relationship '{drug_name} → {target_name}' does not exist in the database. " \
                     f"The cascade prediction is based on hypothetical AI knowledge and may not reflect actual pharmacological interactions. " \
                     f"Please verify the relationship exists before trusting these predictions."
            logger.warning(warning)
        
        # Check for existing cascade
        if not force_repredict:
            existing = self.get_existing_cascade(drug_name, target_name)
            if existing:
                logger.info(f"Using existing cascade for {drug_name} → {target_name}")
                return (existing, warning)
        
        # Use mechanism from database if available
        context_with_mechanism = additional_context
        if mechanism and not additional_context:
            context_with_mechanism = f"Known mechanism: {mechanism}"
        elif mechanism and additional_context:
            context_with_mechanism = f"{additional_context}. Known mechanism: {mechanism}"
        
        # Predict using Gemini API
        cascade = self.predict_cascade_effects(drug_name, target_name, depth, context_with_mechanism)
        
        if not cascade:
            logger.error("Failed to predict cascade")
            return (None, warning)
        
        # Store in Neo4j
        if self.driver:
            success = self.store_cascade_in_neo4j(cascade)
            if not success:
                logger.warning("Failed to store cascade in Neo4j, but returning prediction")
        
        return (cascade, warning)
    
    def batch_predict_cascades(self, drug_target_pairs: List[tuple], depth: int = 2,
                               delay_seconds: int = 1) -> List[Tuple[CascadePrediction, Optional[str]]]:
        """
        Batch predict cascades for multiple drug-target pairs
        
        Args:
            drug_target_pairs: List of (drug_name, target_name) tuples
            depth: Prediction depth
            delay_seconds: Delay between API calls to respect rate limits
        
        Returns:
            List of (CascadePrediction, warning) tuples
        """
        predictions = []
        total = len(drug_target_pairs)
        
        logger.info(f"Starting batch prediction for {total} drug-target pairs")
        
        for i, (drug_name, target_name) in enumerate(drug_target_pairs, 1):
            logger.info(f"Processing {i}/{total}: {drug_name} → {target_name}")
            
            try:
                cascade, warning = self.predict_and_store(drug_name, target_name, depth)
                if cascade:
                    predictions.append((cascade, warning))
                
                # Delay to respect API rate limits (except for last item)
                if i < total:
                    time.sleep(delay_seconds)
                    
            except Exception as e:
                logger.error(f"Error processing {drug_name} → {target_name}: {e}")
                continue
        
        logger.info(f"Batch prediction complete: {len(predictions)}/{total} successful")
        return predictions
    
    def get_cascade_statistics(self) -> Dict[str, Any]:
        """Get statistics about cascade predictions in the database"""
        if not self.driver:
            return {}
        
        try:
            with self.driver.session(database=self.database) as session:
                # Count cascade relationships
                cascade_count = session.run("""
                    MATCH ()-[r:AFFECTS_DOWNSTREAM]->()
                    RETURN count(r) as count
                """).single()['count']
                
                # Count affected entities by type
                entity_counts = session.run("""
                    MATCH ()-[:AFFECTS_DOWNSTREAM]->(e)
                    RETURN labels(e)[0] as entity_type, count(e) as count
                    ORDER BY count DESC
                """).data()
                
                # Average confidence
                avg_confidence = session.run("""
                    MATCH ()-[r:AFFECTS_DOWNSTREAM]->()
                    RETURN avg(r.confidence) as avg_confidence
                """).single()['avg_confidence']
                
                # Unique drug-target pairs with cascades
                unique_pairs = session.run("""
                    MATCH ()-[r:AFFECTS_DOWNSTREAM]->()
                    RETURN count(DISTINCT r.drug_context + ':' + labels(startNode(r))[0]) as count
                """).single()['count']
                
                return {
                    'total_cascade_relationships': cascade_count,
                    'entity_counts_by_type': entity_counts,
                    'average_confidence': round(avg_confidence, 3) if avg_confidence else 0,
                    'unique_drug_target_pairs': unique_pairs
                }
                
        except Exception as e:
            logger.error(f"Error getting cascade statistics: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")


# Example usage
if __name__ == "__main__":
    from config import config as app_config
    
    # Get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        exit(1)
    
    # Initialize predictor
    predictor = BiologicalCascadePredictor(
        gemini_api_key=api_key,
        neo4j_uri=app_config.NEO4J_URI,
        neo4j_user=app_config.NEO4J_USER,
        neo4j_password=app_config.NEO4J_PASSWORD,
        neo4j_database=app_config.NEO4J_DATABASE
    )
    
    try:
        # Create schema (one-time setup)
        print("\n🔧 Creating cascade schema...")
        predictor.create_cascade_schema()
        
        # Example: Predict cascade for Aspirin → COX-2
        print("\n🤖 Predicting cascade effects for Aspirin → COX-2...")
        cascade = predictor.predict_and_store(
            drug_name="aspirin",
            target_name="PTGS2",  # COX-2
            depth=2,
            additional_context="Aspirin irreversibly inhibits COX-2 enzyme"
        )
        
        if cascade:
            print(f"\n✅ Prediction successful!")
            print(f"   Direct effects: {len(cascade.direct_effects)}")
            print(f"   Secondary effects: {len(cascade.secondary_effects)}")
            print(f"   Average confidence: {cascade.total_confidence:.2f}")
            
            print("\n📊 Direct Effects:")
            for effect in cascade.direct_effects[:3]:  # Show first 3
                print(f"   • {effect.entity_name} ({effect.entity_type})")
                print(f"     Effect: {effect.effect_type} (confidence: {effect.confidence:.2f})")
                print(f"     Reasoning: {effect.reasoning[:80]}...")
        
        # Show statistics
        print("\n📈 Cascade Database Statistics:")
        stats = predictor.get_cascade_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
    finally:
        predictor.close()
