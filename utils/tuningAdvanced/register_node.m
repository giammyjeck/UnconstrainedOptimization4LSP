function created = register_node(registry, key, parent_key, reason, cfg, status)
% REGISTER_NODE Aggiunge un nodo al registro dell'albero di ricerca delle
% configurazioni, se non gia' presente.
%
% registry    : containers.Map (KeyType 'char', ValueType 'any'). E' un
%               handle object: le modifiche fatte qui dentro sono visibili
%               anche fuori dalla funzione senza bisogno di riassegnarlo.
% key         : chiave univoca della configurazione (da make_key)
% parent_key  : chiave della configurazione genitrice ('' se e' un nodo
%               radice, cioe' una configurazione della griglia iniziale)
% reason      : stringa che descrive perche' e' stata generata questa
%               configurazione (es. 'bt escalation (bt 5->10)')
% cfg         : struct con i parametri della configurazione
% status      : stato iniziale del nodo ('queued', poi aggiornato con
%               set_status a 'accepted' / 'escalated' / 'discarded')
%
% OUTPUT
% created : true se il nodo e' stato effettivamente creato ora (false se
%           la chiave era gia' presente nel registro)
 
    created = false;
    if ~isKey(registry, key)
        node.parent_key  = parent_key;
        node.reason      = reason;
        node.status      = status;
        node.cfg         = cfg;
        node.short_label = make_short_label(cfg);
        registry(key) = node;
        created = true;
    end
end
 