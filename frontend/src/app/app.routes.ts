import { Routes } from '@angular/router';
import { EpubList } from './pages/epub-list/epub-list';
import { EpubRelationships } from './pages/epub-relationships/epub-relationships';
import { MediaSearch } from './pages/media-search/media-search';


export const routes: Routes = [
  { path: '',         component: MediaSearch },
  { path: 'epubs/:id/relationships', component: EpubRelationships },
];
