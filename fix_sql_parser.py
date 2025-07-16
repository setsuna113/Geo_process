import re

def fix_connection_file():
    # Read the original file
    with open('src/database/connection.py', 'r') as f:
        content = f.read()
    
    # Find the execute_sql_file method
    start_pattern = r'    def execute_sql_file\(self, sql_file: Path\):'
    start_match = re.search(start_pattern, content)
    
    if not start_match:
        print("❌ Could not find execute_sql_file method")
        return False
    
    method_start = start_match.start()
    remaining = content[method_start:]
    
    # Find the next method definition
    next_method_pattern = r'\n    def [a-zA-Z_]'
    next_match = re.search(next_method_pattern, remaining)
    
    if next_match:
        method_end = method_start + next_match.start()
    else:
        # Find end of class or file
        method_end = len(content)
    
    # New implementation
    new_implementation = '''    def execute_sql_file(self, sql_file: Path):
        """Execute SQL file with proper parsing of dollar-quoted strings."""
        if not sql_file.exists():
            raise FileNotFoundError(f"SQL file not found: {sql_file}")
        
        try:
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Smart SQL parsing that handles dollar-quoted strings
            statements = self._parse_sql_statements(sql_content)
            
            with self.get_connection() as conn:
                old_autocommit = conn.autocommit
                conn.autocommit = True
                
                try:
                    with conn.cursor() as cursor:
                        errors = []
                        
                        for i, statement in enumerate(statements):
                            if not statement or statement.strip().startswith('--'):
                                continue
                            
                            try:
                                cursor.execute(statement)
                                logger.debug(f"✅ Executed statement {i+1}/{len(statements)}")
                                
                            except Exception as e:
                                # Skip CREATE INDEX on non-existent tables 
                                if ("does not exist" in str(e) and 
                                    "CREATE INDEX" in statement.upper()):
                                    logger.debug(f"⚠️ Skipping index: {e}")
                                    continue
                                
                                error_msg = f"Statement {i+1} failed: {e}"
                                logger.warning(error_msg)
                                logger.warning(f"Statement: {statement[:200]}...")
                                errors.append((i+1, str(e)))
                        
                        if errors:
                            logger.warning(f"⚠️  {len(errors)}/{len(statements)} statements failed")
                            for stmt_num, error in errors:
                                logger.warning(f"  #{stmt_num}: {error}")
                        
                        logger.info(f"✅ Schema execution completed: {sql_file}")
                            
                finally:
                    conn.autocommit = old_autocommit
            
        except Exception as e:
            logger.error(f"❌ Failed to execute SQL file {sql_file}: {e}")
            raise
    
    def _parse_sql_statements(self, sql_content: str) -> list:
        """Parse SQL content into individual statements, handling dollar-quoted strings."""
        statements = []
        current_statement = ""
        in_dollar_quote = False
        dollar_tag = None
        in_single_quote = False
        in_comment = False
        i = 0
        
        while i < len(sql_content):
            char = sql_content[i]
            
            # Handle line comments
            if char == '-' and i + 1 < len(sql_content) and sql_content[i + 1] == '-':
                # Skip to end of line
                while i < len(sql_content) and sql_content[i] != '\\n':
                    i += 1
                continue
            
            # Handle single quotes (but not in dollar quotes)
            if char == "'" and not in_dollar_quote:
                in_single_quote = not in_single_quote
                current_statement += char
                i += 1
                continue
            
            # Handle dollar-quoted strings
            if char == '$' and not in_single_quote:
                if not in_dollar_quote:
                    # Look for start of dollar quote
                    end_pos = sql_content.find('$', i + 1)
                    if end_pos != -1:
                        potential_tag = sql_content[i:end_pos + 1]
                        # Valid dollar tag (can be $$ or $tag$)
                        if re.match(r'\\$[a-zA-Z0-9_]*\\$', potential_tag):
                            in_dollar_quote = True
                            dollar_tag = potential_tag
                            current_statement += potential_tag
                            i = end_pos + 1
                            continue
                else:
                    # Check for end of dollar quote
                    if sql_content[i:i + len(dollar_tag)] == dollar_tag:
                        in_dollar_quote = False
                        current_statement += dollar_tag
                        i += len(dollar_tag)
                        dollar_tag = None
                        continue
            
            # Handle statement termination
            if (char == ';' and not in_dollar_quote and not in_single_quote):
                current_statement += char
                if current_statement.strip():
                    statements.append(current_statement.strip())
                current_statement = ""
            else:
                current_statement += char
            
            i += 1
        
        # Add final statement if it doesn't end with semicolon
        if current_statement.strip():
            statements.append(current_statement.strip())
        
        return statements

'''
    
    # Replace the method
    new_content = content[:method_start] + new_implementation + content[method_end:]
    
    # Write back to file
    with open('src/database/connection.py', 'w') as f:
        f.write(new_content)
    
    print("✅ Successfully updated connection.py with proper SQL parsing")
    return True

if __name__ == "__main__":
    fix_connection_file()
