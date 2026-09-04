function set_status(registry, key, status)
% SET_STATUS Aggiorna lo stato di un nodo gia' presente nel registro.
% registry e' un containers.Map (handle object), quindi la modifica e'
% visibile anche fuori da questa funzione senza bisogno di riassegnarlo.
%
% status tipici: 'queued', 'accepted', 'escalated', 'discarded'
 
    if isKey(registry, key)
        node = registry(key);
        node.status = status;
        registry(key) = node;
    end
end
 