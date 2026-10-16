import { createRoot } from 'react-dom/client';
import { App } from './App';
import { createClient } from './api/client';
import './styles.css';

const root = createRoot(document.getElementById('root')!);
void createClient().then(client => root.render(<App client={client} />));
