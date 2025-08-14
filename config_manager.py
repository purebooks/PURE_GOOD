#!/usr/bin/env python3
"""
Configuration Management System for Financial Data Cleaner

This module handles loading, validating, and managing user configuration 
preferences for data cleaning and analysis.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration templates and user preferences"""
    
    def __init__(self, templates_dir: str = "config_templates"):
        self.templates_dir = Path(templates_dir)
        self.available_templates = self._discover_templates()
        logger.info(f"Loaded {len(self.available_templates)} configuration templates")
    
    def _discover_templates(self) -> Dict[str, str]:
        """Discover all available configuration templates"""
        templates = {}
        
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory {self.templates_dir} not found")
            return templates
        
        for template_file in self.templates_dir.glob("*.json"):
            template_name = template_file.stem
            templates[template_name] = str(template_file)
            logger.debug(f"Found template: {template_name}")
        
        return templates
    
    def load_config_template(self, template_name: str) -> Dict[str, Any]:
        """Load a configuration template by name"""
        if template_name not in self.available_templates:
            raise ValueError(f"Template '{template_name}' not found. Available: {list(self.available_templates.keys())}")
        
        template_path = self.available_templates[template_name]
        
        try:
            with open(template_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Loaded configuration template: {template_name}")
            return config
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in template {template_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {e}")
            raise
    
    def intent_to_template(self, user_intent: str) -> str:
        """Map user intent to appropriate configuration template"""
        intent_lower = user_intent.lower().strip()
        
        # Direct matches
        intent_mapping = {
            'budget analysis': 'budget_analysis',
            'budget': 'budget_analysis',
            'expense report': 'expense_report',
            'expenses': 'expense_report',
            'subscription audit': 'subscription_audit',
            'subscriptions': 'subscription_audit',
            'clean data': 'standard_clean',
            'standard': 'standard_clean',
        }
        
        # Check direct matches first
        if intent_lower in intent_mapping:
            return intent_mapping[intent_lower]
        
        # Pattern matching for "find X spending"
        if 'find' in intent_lower and ('spending' in intent_lower or 'expenses' in intent_lower):
            return 'find_category_spending'
        
        # Tax-related intents
        if any(word in intent_lower for word in ['tax', 'deduction', 'business']):
            return 'expense_report'
        
        # Default fallback
        logger.info(f"No specific template found for intent '{user_intent}', using standard_clean")
        return 'standard_clean'
    
    def merge_user_preferences(self, template: Dict[str, Any], user_prefs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user preferences with template configuration"""
        if not user_prefs:
            return template.copy()
        
        merged = template.copy()
        
        # Deep merge nested dictionaries
        for key, value in user_prefs.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        
        logger.info(f"Merged user preferences with template")
        return merged
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure and values"""
        required_sections = ['intent', 'cleaning_focus', 'ai_processing']
        
        # Check required sections
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required section: {section}")
                return False
        
        # Validate cleaning focus
        cleaning_focus = config.get('cleaning_focus', {})
        if 'required_columns' not in cleaning_focus:
            logger.error("Missing required_columns in cleaning_focus")
            return False
        
        # Validate AI processing settings
        ai_settings = config.get('ai_processing', {})
        confidence = ai_settings.get('confidence_threshold', 0.7)
        if not (0.0 <= confidence <= 1.0):
            logger.error(f"Invalid confidence_threshold: {confidence}. Must be between 0.0 and 1.0")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def get_config_for_intent(self, user_intent: str, user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get complete configuration for a user intent"""
        
        # Map intent to template
        template_name = self.intent_to_template(user_intent)
        
        # Load base template
        template = self.load_config_template(template_name)
        
        # Apply user preferences if provided
        if user_preferences:
            template = self.merge_user_preferences(template, user_preferences)
        
        # Handle special intent parameters
        template = self._apply_intent_parameters(template, user_intent)
        
        # Validate final configuration
        if not self.validate_config(template):
            logger.warning("Configuration validation failed, using template as-is")
        
        return template
    
    def _apply_intent_parameters(self, config: Dict[str, Any], user_intent: str) -> Dict[str, Any]:
        """Apply intent-specific parameters to configuration"""
        intent_lower = user_intent.lower()
        
        # Extract category from "find X spending" intents
        if 'find' in intent_lower and 'spending' in intent_lower:
            # Extract category (coffee, dining, etc.)
            words = intent_lower.split()
            try:
                find_index = words.index('find')
                spending_index = next(i for i, word in enumerate(words) if 'spending' in word)
                
                if spending_index > find_index + 1:
                    category = ' '.join(words[find_index + 1:spending_index])
                    
                    # Update filtering configuration
                    if 'filtering' not in config:
                        config['filtering'] = {}
                    config['filtering']['target_category'] = category
                    
                    # Add category-specific rules
                    if category in ['coffee']:
                        coffee_vendors = ['starbucks', 'dunkin', 'peets', 'blue bottle', 'cafe']
                        config['categorization']['custom_category_rules'].update({
                            vendor: 'coffee' for vendor in coffee_vendors
                        })
                    
                    logger.info(f"Applied category filter: {category}")
            except (ValueError, IndexError):
                pass
        
        return config
    
    def list_available_templates(self) -> List[Dict[str, str]]:
        """Get list of available templates with descriptions"""
        templates_info = []
        
        for template_name in self.available_templates:
            try:
                config = self.load_config_template(template_name)
                templates_info.append({
                    'name': template_name,
                    'intent': config.get('intent', template_name),
                    'description': config.get('description', 'No description available')
                })
            except Exception as e:
                logger.warning(f"Could not load template {template_name}: {e}")
        
        return templates_info

# Global instance for easy importing
config_manager = ConfigManager()

def get_config_for_intent(user_intent: str, user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function to get configuration for an intent"""
    return config_manager.get_config_for_intent(user_intent, user_preferences)

def list_templates() -> List[Dict[str, str]]:
    """Convenience function to list available templates"""
    return config_manager.list_available_templates()

if __name__ == "__main__":
    # Test the configuration manager
    import pprint
    
    print("🧪 Testing Configuration Manager")
    print("=" * 50)
    
    # Test available templates
    templates = list_templates()
    print(f"\n📋 Available Templates ({len(templates)}):")
    for template in templates:
        print(f"  - {template['name']}: {template['description']}")
    
    # Test intent mapping
    test_intents = [
        "budget analysis",
        "find coffee spending", 
        "expense report",
        "subscription audit",
        "something random"
    ]
    
    print(f"\n🎯 Intent Mapping Tests:")
    for intent in test_intents:
        config = get_config_for_intent(intent)
        print(f"  '{intent}' → {config['intent']} template")
    
    # Test configuration structure
    print(f"\n📊 Sample Configuration (budget_analysis):")
    budget_config = get_config_for_intent("budget analysis")
    pprint.pprint(budget_config, width=100, depth=2)