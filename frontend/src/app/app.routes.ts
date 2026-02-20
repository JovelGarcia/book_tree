import { Routes } from '@angular/router';
import { EpubListComponent } from './pages/epub-list/epub-list';
import { EpubDetail } from './pages/epub-detail/epub-detail';
import { Upload } from './pages/upload/upload';
import { ConfirmDelete } from './pages/confirm-delete/confirm-delete';

export const routes: Routes = [
  { path: '', component: EpubListComponent },
  { path: 'epub/:id', component: EpubDetail },
  { path: 'upload', component: Upload },
  { path: 'delete/:id', component: ConfirmDelete },
];
