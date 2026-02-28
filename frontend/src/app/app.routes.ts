import { Routes } from '@angular/router';
import { EpubList } from './pages/epub-list/epub-list';
import { EpubRelationships } from './pages/epub-relationships/epub-relationships';

export const routes: Routes = [
  { path: '',         component: EpubList },
  { path: 'epubs/:id/relationships', component: EpubRelationships },
];
