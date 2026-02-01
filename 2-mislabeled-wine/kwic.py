import pandas as pd
from itertools import islice
from IPython.core.display import display, HTML
import re

def kwic(pattern, text, k=75, max_n=10): 
    """Display keyword in context concordance in a notebook"""    
    if isinstance(text, pd.Series):        
        it = text.iteritems()    
    else:        
        it = enumerate(text)       
    pattern = r'\b' + pattern + r'\b'    
    html = ['<table>'] + list(islice(_kwic_rows(pattern, it, k), max_n)) + ['</table>']
    display(HTML(''.join(html)))
    
def _kwic_rows(pattern, it, k):    
    for i,t in it:        
        for m in re.finditer(pattern, t, re.I):            
            s, e = m.span()            
            yield f'''<tr><td>{i}</td>                      <td style="text-align:right;white-space:nowrap">{t[max(0,s-k):s]:>{k}}</td>                      <td style="text-align:center;white-space:nowrap">{t[s:e]}</td>                      <td style="text-align:left;white-space:nowrap">{t[e:e+k]:{k}}</td></tr>'''