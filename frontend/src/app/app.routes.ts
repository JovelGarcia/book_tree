import { Routes } from '@angular/router';
import { MediaGraph } from './pages/media-graph/media-graph';
import { MediaSearch } from './pages/media-search/media-search';


export const routes: Routes = [
  { path: '',         component: MediaSearch },
  { path: 'media/:id/graph', component: MediaGraph },
];
